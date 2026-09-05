from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"


def main() -> None:
    print("Loading IndicTrans2 model...")

    # ---------------------------------------------------------
    # 1. Load tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # ---------------------------------------------------------
    # 3. Select device
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    print("✓ Model loaded")
    print(f"✓ Device: {device}")

    # ---------------------------------------------------------
    # 4. IndicTrans2 processor
    # ---------------------------------------------------------
    processor = IndicProcessor(inference=True)

    # ---------------------------------------------------------
    # 5. Test sentences
    # ---------------------------------------------------------
    texts = [
        "मशीन लर्निंग डेटा से पैटर्न सीखता है।",
        "इस वीडियो में हम LangChain के बारे में बात करेंगे।",
        "आप इस तकनीक का उपयोग अपने प्रोजेक्ट में कर सकते हैं।",
    ]

    print("\n" + "=" * 70)
    print("INDICTRANS2 TRANSLATION TEST")
    print("=" * 70)

    for index, text in enumerate(texts, start=1):

        print(f"\n--- Sentence {index} ---")

        print("\nHindi:")
        print(text)

        # -----------------------------------------------------
        # Preprocess
        # -----------------------------------------------------
        batch = processor.preprocess_batch(
            [text],
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
        )

        # -----------------------------------------------------
        # Tokenize
        # -----------------------------------------------------
        inputs = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # -----------------------------------------------------
        # Generate
        # -----------------------------------------------------
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=256,
                num_beams=1,
                use_cache=False,
            )

        # -----------------------------------------------------
        # Decode
        # -----------------------------------------------------
        generated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # -----------------------------------------------------
        # Postprocess
        # -----------------------------------------------------
        translations = processor.postprocess_batch(
            generated_text,
            lang=TGT_LANG,
        )

        translation = translations[0]

        print("\nEnglish:")
        print(translation)

    print("\n" + "=" * 70)
    print("INDICTRANS2 TEST COMPLETED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()