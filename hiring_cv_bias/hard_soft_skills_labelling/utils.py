import polars as pl
import transformers
from tqdm.notebook import tqdm


def batch_classify_skills(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    skills: list[str],
    batch_size: int,
) -> list[str]:
    labels = []
    for i in tqdm(range(0, len(skills), batch_size)):
        batch = skills[i : i + batch_size]
        prompts = [
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        Classify the following skill as either "Hard" or "Soft". Think step-by-step about what the skill involves before giving your answer. If you are unsure or do not know the classification, respond with only the word "Unknown". After your reasoning, respond with only one word: ‘Hard’, ‘Soft’ or ‘Unknown’
        Example 1:
        Skill: Data Analysis
        Step 1: Data Analysis involves technical ability to work with data, use software tools, and interpret numerical results.
        Step 2: These are measurable, teachable skills requiring specific knowledge.
        Answer: Hard
        
        Example 2:
        Skill: Communication
        Step 1: Communication involves interpersonal skills, expressing ideas clearly, and understanding others.
        Step 2: These skills are more about behavior and attitude rather than technical knowledge.
        Answer: Soft

        Example 3:
        Skill: Financial Modeling 
        Step 1: Financial Modeling involves technical ability to build spreadsheet models, perform valuations, and analyze financial statements using specific formulas and methodologies. 
        Step 2: These are measurable, teachable skills requiring knowledge of accounting principles, Excel functions, and financial theory. 
        Answer: Hard
        
        Example 4:
        Skill: Time Management
        Step 1: Time Management refers to organizing one's own schedule and priorities effectively.
        Step 2: This is a behavioral skill that improves productivity and is not technical.
        Answer: Soft
        
        Now classify this skill:
        Skill: {skill}
        Step 1: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            for skill in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in decoded:
            print(text)
            label = text.split()[-1]
            labels.append(label)
    return labels


def clean_results(results: pl.DataFrame) -> pl.DataFrame:
    return results.with_columns(
        pl.when(~pl.col("label").is_in(["Hard", "Soft", "Unknown"]))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("label"))
        .alias("label")
    )
