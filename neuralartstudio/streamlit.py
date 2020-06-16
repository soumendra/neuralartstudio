import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .image_style import ImageStyle


def neural_style_transfer(contentpath: str, stylepath: str):
    nst = ImageStyle(contentpath=contentpath, stylepath=stylepath)

    style_weight = st.sidebar.number_input(
        "Choose style weight", min_value=0, value=nst.style_weight
    )
    content_weight = st.sidebar.number_input(
        "Choose content weight", min_value=0, value=nst.content_weight
    )
    num_steps = st.sidebar.number_input(
        "No of steps (iterations)", min_value=1, value=nst.num_steps
    )
    input_init = st.sidebar.radio(
        "Input image init", ("From content image", "Gaussian noise")
    )

    uploaded_content_image = st.sidebar.file_uploader(
        "Upload content image", type=["png", "jpg", "jpeg"]
    )
    if uploaded_content_image is not None:
        nst.content_img = nst.read_image(uploaded_content_image)

    if input_init == "Gaussian noise":
        nst.input_init(strategy="noise")
    elif input_init == "From content image":
        nst.input_init(strategy="content")

    plt_, imgs = nst.plotobj()
    st.pyplot(fig=plt_, clear_figure=True)

    if st.sidebar.button("Train!"):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        chart_raw_losses = st.line_chart()
        nst.train(
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            streamlit={
                "status_text": status_text,
                "progress_bar": progress_bar,
                "chart_raw_losses": chart_raw_losses,
            },
        )
        plt_, imgs = nst.plotobj()
        st.write("After training ...")
        st.pyplot(fig=plt_, clear_figure=True)

        if num_steps > 20:
            mode = "lines"
        else:
            mode = "lines+markers"

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["style_loss"],
                mode=mode,
                name="style_loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["content_loss"],
                mode=mode,
                name="content_loss",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["total_loss"],
                mode=mode,
                name="total_loss",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["style_loss/content_loss"],
                mode=mode,
                name="style/content loss",
            ),
            row=4,
            col=1,
        )
        fig.update_xaxes(title_text="Steps", row=4, col=1)
        fig.update_layout(
            title="Raw losses",
            font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"),
            height=400,
            width=800,
        )
        st.plotly_chart(figure_or_data=fig)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["style_loss_weighted"],
                mode=mode,
                name="style_loss_weighted",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["content_loss_weighted"],
                mode=mode,
                name="content_loss_weighted",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["total_loss_weighted"],
                mode=mode,
                name="total_loss_weighted",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=nst.logMetrics["steps"],
                y=nst.logMetrics["style_loss/content_loss (weighted)"],
                mode=mode,
                name="style/content loss (weighted)",
            ),
            row=4,
            col=1,
        )
        fig.update_xaxes(title_text="Steps", row=4, col=1)
        fig.update_layout(
            title="Weighted losses",
            font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"),
            height=400,
            width=800,
        )
        st.plotly_chart(figure_or_data=fig)
        st.write(nst.logMetrics)
