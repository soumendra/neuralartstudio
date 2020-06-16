import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLossLayer(nn.Module):
    def __init__(
        self, target,
    ):
        super(ContentLossLayer, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLossLayer(nn.Module):
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=1 (batch size), b=#feature maps
        # (c,d)=dimensions of a feature map (N=c*d)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def __init__(self, target_feature):
        super(StyleLossLayer, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ImageStyle:
    def __init__(self, stylepath: str, contentpath: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.imsize = 512
        else:
            self.device = torch.device("cpu")
            self.imsize = 128

        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
        self.style_weight = 1000000
        self.content_weight = 1
        self.num_steps = 3

        self.stylepath = stylepath
        self.contentpath = contentpath
        self.style_img = self.read_image(self.stylepath)
        self.content_img = self.read_image(self.contentpath)
        self.input_img = None

    def input_init(self, strategy="content"):
        if strategy == "noise":
            self.input_img = torch.randn(
                self.content_img.data.size(), device=self.device
            )
        elif strategy == "content":
            self.input_img = self.content_img.clone()
        else:
            raise ValueError(f"Unknown strategy value: {strategy}")
        return self

    def read_image(self, impath):
        image = Image.open(impath)
        trfm = transforms.Compose(
            [
                transforms.CenterCrop(min(image.size)),
                transforms.Resize(self.imsize),
                transforms.ToTensor(),
            ]
        )
        image = trfm(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imshow(self, tensor):
        image = tensor.cpu().clone().squeeze(0)
        image = transforms.ToPILImage()(image)
        return image

    def save(self):
        style = self.imshow(self.style_img)
        content = self.imshow(self.content_img)
        input_ = self.imshow(self.input_img)  # torchvision.utils.save_image()?
        style.save("style.png")
        content.save("content.png")
        input_.save("input.png")
        return self

    def plotobj(self):
        nrows, ncols = 1, 3

        style = self.imshow(self.style_img)
        content = self.imshow(self.content_img)
        input_ = self.imshow(self.input_img)
        imgs = [style, content, input_]
        labels = ["Style image", "Content image", "Input image"]

        plt = self.plot_array(imgs, labels, nrows, ncols, figsize_factor=4)
        return plt, imgs

    def plot(self, filename="default.png"):
        plt, _ = self.plotobj()
        plt.savefig(filename)

    def plot_array(self, img_array, label_array, nrows, ncols, figsize_factor=4):
        figsize = [ncols * figsize_factor, nrows * figsize_factor]
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, axi in enumerate(ax.flat):
            # i runs from 0 to (nrows*ncols-1)
            # axi is equivalent with ax[rowid][colid]
            # get indices of row/column: rowid = i // ncols; colid = i % ncols
            # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
            axi.imshow(img_array[i])
            axi.set_title(label_array[i])
            axi.axis("off")
        return plt

    def get_model_and_losses(self):
        cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        cnn = copy.deepcopy(cnn)
        content_losses = []
        style_losses = []
        normalization = Normalization(
            self.normalization_mean, self.normalization_std
        ).to(self.device)
        model = nn.Sequential(normalization)

        conv_block_idx = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                conv_block_idx += 1
                name = f"conv_{conv_block_idx}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{conv_block_idx}"
                # The in-place version doesn't play very nicely with the ContentLoss and StyleLoss
                # we insert below. So we replace with out-of-place ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{conv_block_idx}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{conv_block_idx}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(self.content_img).detach()
                content_loss = ContentLossLayer(target)
                model.add_module(f"content_loss_{conv_block_idx}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLossLayer(target_feature)
                model.add_module(f"style_loss_{conv_block_idx}", style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLossLayer) or isinstance(
                model[i], StyleLossLayer
            ):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([self.input_img.requires_grad_()])
        return optimizer

    def train(
        self, num_steps=None, style_weight=None, content_weight=None, streamlit=None
    ):
        if style_weight is not None:
            self.style_weight = style_weight
        if content_weight is not None:
            self.content_weight = content_weight
        if num_steps is not None:
            self.num_steps = num_steps
        column_names = [
            "style_loss",
            "content_loss",
            "total_loss",
            "style_loss/content_loss",
            "style_loss_weighted",
            "content_loss_weighted",
            "total_loss_weighted",
            "style_loss/content_loss (weighted)",
        ]
        self.logMetrics = pd.DataFrame(columns=column_names)

        model, style_losses, content_losses = self.get_model_and_losses()
        optimizer = self.get_input_optimizer()

        for i in range(num_steps):
            print(f"############# {num_steps}")

            def closure():
                self.input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                logMetricsDict = {
                    "style_loss": style_score.item(),
                    "content_loss": content_score.item(),
                    "total_loss": (style_score + content_score).item(),
                    "style_loss/content_loss": (style_score / content_score).item(),
                }

                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()

                logMetricsDict.update(
                    {
                        "style_loss_weighted": style_score.item(),
                        "content_loss_weighted": content_score.item(),
                        "total_loss_weighted": (style_score + content_score).item(),
                        "style_loss/content_loss (weighted)": (
                            style_score / content_score
                        ).item(),
                    }
                )

                self.logMetrics.loc[
                    0
                    if pd.isnull(self.logMetrics.index.max())
                    else self.logMetrics.index.max() + 1
                ] = logMetricsDict

                if streamlit is not None:
                    streamlit["status_text"].text(
                        f"{int(100*(i+1)/num_steps)}% Complete"
                    )
                    streamlit["progress_bar"].progress(int(100 * (i + 1) / num_steps))
                    update_df = pd.DataFrame(data=logMetricsDict, index=[i])
                    streamlit["chart_raw_losses"].add_rows(
                        update_df[["total_loss_weighted"]]
                    )

                print(f"Run: {i}")
                print(f"Style Loss : {style_score.item():.6f}")
                print(f"Content Loss: {content_score.item():.6f}")
                print(f"Total Loss: {(style_score+content_score).item():.6f}\n")

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)
        self.logMetrics["steps"] = list(
            range(1, len(self.logMetrics["style_loss"]) + 1)
        )

        return self


if __name__ == "__main__":
    nst = ImageStyle(
        contentpath="./assets/selfie.png", stylepath="./assets/picasso.jpg"
    )
    nst.plot("before_training.png")
    # nst.train(num_steps=400, style_weight=1000000, content_weight=1)
    # nst.plot("after_training.png")
    nst.save()
