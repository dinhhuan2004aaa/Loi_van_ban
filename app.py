import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "tranformer_model\\tranformer"  
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cpu")  

def correct_grammar(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4)
    corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return corrected_text

iface = gr.Interface(
    fn=correct_grammar,
    inputs=gr.Textbox(lines=3, placeholder="Nhập câu cần sửa..."),
    outputs="text",
    title=" Sửa lỗi ngữ pháp (Grammar Correction)",
    description="Nhập câu tiếng Anh sai cú pháp vào ô bên dưới, mô hình sẽ gợi ý câu đúng.",
    examples=[
        "He go to school yesterday.",
        "She are a good teacher.",
        "I has a pen.",
    ]
)

iface.launch()
