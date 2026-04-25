%%writefile cap_inf.py
import pandas as pd
import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path


def eval_model(args):
    disable_torch_init()

    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, _, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor

    # Read CSV
    df = pd.read_csv(os.path.expanduser(args.question_file))
    df = df.iloc[:1000]

    print("Loaded:", len(df))
    print(df.head())

    # Prepare caption records
    captions = []
    for _, row in df.iterrows():
        captions.append({
            "image": row["image"],
            "gt_caption": row["caption"] if "caption" in row else ""
        })

    # Output file
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")

    # Run in batches
    for i in tqdm(range(0, len(captions), args.batch_size)):
        input_batch = []
        image_batch = []
        valid_indices = []

        batch_end = min(i + args.batch_size, len(captions))

        for j in range(i, batch_end):
            img_path = os.path.join(args.image_folder, captions[j]["image"])

            # Check if image exists
            if not os.path.exists(img_path):
                print(f"[WARNING] File not found: {img_path}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Could not open {img_path}: {e}")
                continue

            # Caption prompt
            qs = (
                "Generate a detailed caption describing all visible elements in the satellite image, "
                "including object types, counts, relative locations, and scene context."
            )

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n"chatgpt openai gpt 5  ai creativity ai literacy ai_assistant
chatgpt openai gpt 5  ai creativity ai literacy ai_assistant
  + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            input_batch.append(input_ids)
            image_batch.append(img)
            valid_indices.append(j)

        if len(valid_indices) == 0:
            continue

        # Pad text inputs
        max_len = max(x.size(1) for x in input_batch)
        final_inputs = [
            torch.cat(
                (torch.zeros((1, max_len - t.size(1)), dtype=t.dtype, device=t.get_device()), t),
                dim=1
            )
            for t in input_batch
        ]
        final_input_tensors = torch.cat(final_inputs, dim=0)

        # Process images
        image_tensor_batch = image_processor.preprocess(
            image_batch,
            crop_size={'height': 504, 'width': 504},
            size={'shortest_edge': 504},
            return_tensors='pt'
        )['pixel_values']

        # Model inference
        with torch.inference_mode():
            output_ids = model.generate(
                final_input_tensors,
                images=image_tensor_batch.half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=256,
                num_beams=1,
            )

        input_len = final_input_tensors.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

        # Write results
        for out, idx in zip(outputs, valid_indices):
            ans_file.write(json.dumps({
                "image": captions[idx]["image"],
                "pred_caption": out.strip(),
                "gt_caption": captions[idx]["gt_caption"]
            }) + "\n")
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="SkySenseGPT-7B-CLIP-ViT")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="table.csv")
    parser.add_argument("--answers-file", type=str, default="captions.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    eval_model(args)
