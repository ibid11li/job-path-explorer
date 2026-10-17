import pandas as pd
import gradio as gr

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================
# تحميل البيانات
# =========================
df = pd.read_csv("Jadarat_data.csv")


# =========================
# تنظيف بسيط للبيانات
# =========================
df = df.drop_duplicates()

df["job_title"] = df["job_title"].fillna("").str.strip()
df["job_desc"] = df["job_desc"].fillna("").str.strip()
df["job_tasks"] = df["job_tasks"].fillna("").str.strip()


# =========================
# اختيار الأعمدة المهمة فقط
# =========================
jobs = df[
    [
        "job_title",
        "job_desc",
        "job_tasks"
    ]
].copy()


# =========================
# تحميل نموذج Embeddings
# =========================
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# حساب Embeddings للوظائف مرة واحدة
# =========================
title_embeddings = model.encode(
    jobs["job_title"].tolist()
)

tasks_embeddings = model.encode(
    jobs["job_tasks"].tolist()
)

desc_embeddings = model.encode(
    jobs["job_desc"].tolist()
)


# =========================
# دالة اقتراح الوظائف
# =========================
def recommend_jobs(user_text):

    try:
        if not user_text or not user_text.strip():
            return "اكتب وصفًا للأشياء التي تحب القيام بها أولًا."

        # تحويل وصف المستخدم إلى Embedding
        user_embedding = model.encode(
            [user_text]
        )

        # حساب التشابه مع المسمى
        title_similarity = cosine_similarity(
            user_embedding,
            title_embeddings
        )[0]

        # حساب التشابه مع المهام
        tasks_similarity = cosine_similarity(
            user_embedding,
            tasks_embeddings
        )[0]

        # حساب التشابه مع الوصف
        desc_similarity = cosine_similarity(
            user_embedding,
            desc_embeddings
        )[0]

        # الدرجة النهائية
        jobs["final_similarity"] = (
            title_similarity * 0.30
            + tasks_similarity * 0.50
            + desc_similarity * 0.20
        )

        # ترتيب الوظائف
        top_jobs = jobs.sort_values(
            by="final_similarity",
            ascending=False
        )

        # إزالة تكرار نفس المسمى
        top_jobs = top_jobs.drop_duplicates(
            subset=["job_title"]
        )

        # أفضل 5
        top_5 = top_jobs.head(5)

        # تجهيز النتائج
        results = ""

        for number, (_, row) in enumerate(
            top_5.iterrows(),
            start=1
        ):
            job_title = row["job_title"]

            score = round(
                row["final_similarity"] * 100,
                2
            )

            results += (
                f"{number}. {job_title} ({score}%)\n"
            )

        return results

    except Exception as error:
        return f"حدث خطأ: {error}"


# =========================
# تنسيق الواجهة
# =========================
css = """
.gradio-container {
    max-width: 850px !important;
    margin: auto !important;
}

#title {
    text-align: center;
    direction: rtl;
}

#description {
    text-align: center;
    direction: rtl;
    margin-bottom: 25px;
}

#user-input textarea {
    direction: rtl !important;
    text-align: right !important;
    font-size: 16px !important;
}

#user-input label {
    direction: rtl !important;
    text-align: right !important;
}

#results textarea {
    direction: rtl !important;
    text-align: right !important;
    font-size: 17px !important;
    line-height: 2 !important;
}

#results label {
    direction: rtl !important;
    text-align: right !important;
}

#submit-button {
    width: 100%;
}

@media (max-width: 600px) {
    .gradio-container {
        padding: 12px !important;
    }

    #title h1 {
        font-size: 26px !important;
    }

    #user-input textarea {
        min-height: 150px !important;
    }
}
"""


# =========================
# واجهة التطبيق
# =========================
with gr.Blocks(css=css) as app:

    gr.Markdown(
        "# مستكشف المسار الوظيفي",
        elem_id="title"
    )

    gr.Markdown(
        "اكتب الأشياء التي تحب القيام بها، وسنقترح لك أقرب المسميات الوظيفية.",
        elem_id="description"
    )

    user_input = gr.Textbox(
        label="اكتب الأشياء اللي تحب تسويها",
        placeholder="مثال: أحب تحليل البيانات والعمل مع قواعد البيانات وحل المشاكل",
        lines=6,
        elem_id="user-input"
    )

    submit_button = gr.Button(
        "اقترح لي وظائف",
        variant="primary",
        elem_id="submit-button"
    )

    results = gr.Textbox(
        label="الوظائف المقترحة",
        lines=8,
        interactive=False,
        elem_id="results"
    )

    submit_button.click(
        fn=recommend_jobs,
        inputs=user_input,
        outputs=results
    )


# =========================
# تشغيل التطبيق
# =========================
app.launch()