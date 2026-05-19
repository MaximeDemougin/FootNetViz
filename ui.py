import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --footnet-ink: #10233f;
                --footnet-muted: #5e6d82;
                --footnet-accent: #0ea5a4;
                --footnet-accent-soft: rgba(14, 165, 164, 0.12);
                --footnet-surface: rgba(255, 255, 255, 0.88);
                --footnet-border: rgba(16, 35, 63, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 164, 0.18), transparent 30%),
                    radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 26%),
                    linear-gradient(180deg, #eef7f6 0%, #f7fafc 42%, #fbf7ef 100%);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #12233f 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            .footnet-hero {
                padding: 1.4rem 1.6rem;
                border-radius: 24px;
                background:
                    linear-gradient(135deg, rgba(16, 35, 63, 0.96), rgba(14, 165, 164, 0.78)),
                    linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent);
                color: white;
                box-shadow: 0 24px 50px rgba(16, 35, 63, 0.16);
                margin-bottom: 1.2rem;
            }

            .footnet-hero__eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                opacity: 0.75;
                margin-bottom: 0.45rem;
            }

            .footnet-hero h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.05;
            }

            .footnet-hero p {
                margin: 0.55rem 0 0;
                max-width: 48rem;
                font-size: 0.98rem;
                color: rgba(255, 255, 255, 0.88);
            }

            [data-testid="stMetric"] {
                border: 1px solid var(--footnet-border);
                background: var(--footnet-surface);
                border-radius: 18px;
                padding: 0.75rem 0.9rem;
                box-shadow: 0 12px 28px rgba(16, 35, 63, 0.05);
            }

            .footnet-panel {
                border: 1px solid var(--footnet-border);
                background: var(--footnet-surface);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 30px rgba(16, 35, 63, 0.05);
            }

            .footnet-note {
                color: var(--footnet-muted);
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, eyebrow: str = "FootNetViz") -> None:
    st.markdown(
        f"""
        <section class="footnet-hero">
            <div class="footnet-hero__eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )