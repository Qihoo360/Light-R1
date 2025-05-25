import re
from rouge_score import rouge_scorer

def ends_with_repeating_substring(s, min_repetition=5):
    for i in range(1, len(s)):
        substring = s[-i:]
        if substring * min_repetition == s[-len(substring) * min_repetition:]:
            return True
    return False

def compute_repeat_penalty(solution_str, tail_token_num=2000):
    """
    Computes the repeat penalty for a given solution string.
    """
    char_score = 1.0 if ends_with_repeating_substring(solution_str, min_repetition=5) else 0.0
    
    # ROUGE scorer 준비    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize phrase_score and paragraph_score to handle cases where they might not be assigned
    phrase_score = 0.0
    paragraph_score = 0.0
    
    # -------------------------------
    # 1. Character-level repetition using your function
    # 2~3. Phrase-level & Paragraph-level with ROUGE on <think> sections
    # -------------------------------
    if "<think>" in solution_str and "</think>" in solution_str:
        
        think_sections = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
        full_think = " ".join(think_sections).strip() if think_sections else ""
        tail_text = full_think[-tail_token_num:]  # token 수 기준이지만 char approximation
    
        # 단락 분리 (빈 줄 기준)
        paragraphs = [p.strip() for p in tail_text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return char_score, phrase_score, paragraph_score
        
        # paragraph의 끝에서 character-level repetition 한 번 더 확인
        if char_score == 0.0:
            char_score = 1.0 if ends_with_repeating_substring(paragraphs[-1], min_repetition=5) else 0.0        
    
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
    
        # 모든 단락 쌍 간 ROUGE 비교
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                scores = scorer.score(paragraphs[i], paragraphs[j])
                rouge1_scores.append(scores["rouge1"].fmeasure)
                rouge2_scores.append(scores["rouge2"].fmeasure)
                rougeL_scores.append(scores["rougeL"].fmeasure)
    
        # 평균 반복 score (높을수록 반복)
        phrase_score = (sum(rouge1_scores) + sum(rouge2_scores)) / (2 * len(rouge1_scores) + 1e-6)
        paragraph_score = sum(rougeL_scores) / (len(rougeL_scores) + 1e-6)

    ### </think> 이후 텍스트 추출
    if "</think>" in solution_str:
        post_think_text = re.split(r"</think>", solution_str, maxsplit=1)[-1].strip()
        post_think_tail_text = post_think_text[-tail_token_num:]
        post_think_paragraphs = [p.strip() for p in post_think_tail_text.split('\n\n') if p.strip()]

        if len(post_think_paragraphs) >= 2:
            post_rouge1_scores = []
            post_rouge2_scores = []
            post_rougeL_scores = []

            for i in range(len(post_think_paragraphs)):
                for j in range(i + 1, len(post_think_paragraphs)):
                    scores = scorer.score(post_think_paragraphs[i], post_think_paragraphs[j])
                    post_rouge1_scores.append(scores["rouge1"].fmeasure)
                    post_rouge2_scores.append(scores["rouge2"].fmeasure)
                    post_rougeL_scores.append(scores["rougeL"].fmeasure)

            phrase_score_post = (sum(post_rouge1_scores) + sum(post_rouge2_scores)) / (2 * len(post_rouge1_scores) + 1e-6)
            paragraph_score_post = sum(post_rougeL_scores) / (len(post_rougeL_scores) + 1e-6)

            # Weighted combination of scores
            if phrase_score == 0.0 and paragraph_score == 0.0:
                phrase_score = phrase_score_post
                paragraph_score = paragraph_score_post
            else:
                phrase_score = (phrase_score + phrase_score_post) / 2
                paragraph_score = (paragraph_score + paragraph_score_post) / 2

    else:
        ### </think>가 없는 경우 추출
        # Check repetition in the entire text if no </think> sections are found
        pre_tail_text = solution_str[-tail_token_num:]
        
        pre_paragraphs = [p.strip() for p in pre_tail_text.split('\n\n') if p.strip()]

        if len(pre_paragraphs) >= 2:
            pre_rouge1_scores = []
            pre_rouge2_scores = []
            pre_rougeL_scores = []

            for i in range(len(pre_paragraphs)):
                for j in range(i + 1, len(pre_paragraphs)):
                    scores = scorer.score(pre_paragraphs[i], pre_paragraphs[j])
                    pre_rouge1_scores.append(scores["rouge1"].fmeasure)
                    pre_rouge2_scores.append(scores["rouge2"].fmeasure)
                    pre_rougeL_scores.append(scores["rougeL"].fmeasure)

            phrase_score_pre = (sum(pre_rouge1_scores) + sum(pre_rouge2_scores)) / (2 * len(pre_rouge1_scores) + 1e-6)
            paragraph_score_pre = sum(pre_rougeL_scores) / (len(pre_rougeL_scores) + 1e-6)

            # Average scores across all scenarios
            if phrase_score == 0.0 and paragraph_score == 0.0:
                phrase_score = phrase_score_pre
                paragraph_score = paragraph_score_pre
            else:
                phrase_score = (phrase_score + phrase_score_pre) / 2
                paragraph_score = (paragraph_score + paragraph_score_pre) / 2

    return char_score, phrase_score, paragraph_score