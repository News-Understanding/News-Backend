import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarize_text(text):
    def batch_generator(data, batch_size):
        s = 0
        e = s + batch_size
        while s < len(data):
            yield data[s:e]
            s = e
            e = min(s + batch_size, len(data))

    def summarize_with_model(model_checkpoint, articles, batch_size=8):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        def perform_inference(batch):
            inputs = tokenizer(
                batch,
                max_length=1024,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            summary_ids = model.generate(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                num_beams=2,
                max_length=1024,
                early_stopping=True,
            )
            return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        res = []
        summary_text = "summarize: " + text
        res += perform_inference([summary_text])

        # Clean up
        del tokenizer
        del model
        torch.cuda.empty_cache()

        return res[0]

    # t5_small_summary = summarize_with_model("t5-small", [text])
    # t5_base_summary = summarize_with_model("t5-base", [text])
    # flan_t5_base_summary = summarize_with_model("google/flan-t5-base", [text])
    bart_large_summary = summarize_with_model("facebook/bart-large-cnn", [text])

    return bart_large_summary
        # flan_t5_base_summary
        # bart_large_summary
    # t5_small_summary,
    #
    # flan_t5_base_summary,
