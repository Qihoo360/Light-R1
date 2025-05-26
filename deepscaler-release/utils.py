import re
from rouge_score import rouge_scorer

def ends_with_repeating_substring(s, min_repetition=5):
    for i in range(1, len(s)):
        substring = s[-i:]
        if substring * min_repetition == s[-len(substring) * min_repetition:]:
            return True
    return False

def compute_internal_rouge_score(paragraph, scorer, phrase_len=5):
    """
    Computes the average pairwise ROUGE-1, ROUGE-2, and ROUGE-L f1 scores among sliding window phrases
    within a single paragraph. Returns a combined repetition score.
    """
    words = paragraph.split()
    if len(words) < phrase_len * 2:
        return 0.0  # 너무 짧은 문단은 비교하지 않음

    # 슬라이딩 윈도우로 phrase 생성
    phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words) - phrase_len + 1)]

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            scores = scorer.score(phrases[i], phrases[j])
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

    if not rouge1_scores:
        return 0.0

    # 평균 ROUGE 반복 점수 계산
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    # 세 점수의 평균으로 최종 repetition score 반환 (가중 평균 가능)
    combined_score = (avg_rouge1 + avg_rouge2 + avg_rougeL) / 3
    return combined_score

def compute_external_rouge_score(paragraphs, scorer):
    """
    Computes the pairwise ROUGE-1, ROUGE-2, and ROUGE-L f1 scores among paragraphs which are splitted by "\n\n".
    It returns a combined repetition score.
    """
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
    return phrase_score, paragraph_score

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

        # 마지막 문단에서 내부 phrase 체크 한 번 더 하기
        internal_phrase_score = compute_internal_rouge_score(paragraph=paragraphs[-1], scorer=scorer, phrase_len=5)
        external_phrase_score, paragraph_score = compute_external_rouge_score(paragraphs=paragraphs, scorer=scorer)

        phrase_score = (external_phrase_score + internal_phrase_score) / 2

        # </think> 이후 텍스트 추출
        post_think_text = re.split(r"</think>", solution_str, maxsplit=1)[-1].strip()
        post_think_tail_text = post_think_text[-tail_token_num:]
        post_think_paragraphs = [p.strip() for p in post_think_tail_text.split('\n\n') if p.strip()]
        
        if len(post_think_paragraphs) >= 2:
            
            # 마지막 문단에서 내부 phrase 체크 한 번 더 하기
            internal_phrase_score_post = compute_internal_rouge_score(paragraph=post_think_paragraphs[-1], scorer=scorer, phrase_len=5)
            phrase_score_post, paragraph_score_post = compute_external_rouge_score(paragraphs=post_think_paragraphs, scorer=scorer)
            
            # Weighted combination of scores
            if phrase_score == 0.0 and paragraph_score == 0.0:
                phrase_score = (phrase_score_post + internal_phrase_score_post) / 2
                paragraph_score = paragraph_score_post
            else:
                phrase_score_post = (phrase_score_post + internal_phrase_score_post) / 2
                phrase_score = (phrase_score + phrase_score_post) / 2
                paragraph_score = (paragraph_score + paragraph_score_post) / 2
    
    ### </think> 이후 텍스트 추출
    if "</think>" in solution_str:
        post_think_text = re.split(r"</think>", solution_str, maxsplit=1)[-1].strip()
        post_think_tail_text = post_think_text[-tail_token_num:]
        post_think_paragraphs = [p.strip() for p in post_think_tail_text.split('\n\n') if p.strip()]
        
        if len(post_think_paragraphs) >= 2:
            
            # 마지막 문단에서 내부 phrase 체크 한 번 더 하기
            internal_phrase_score_post = compute_internal_rouge_score(paragraph=post_think_paragraphs[-1], scorer=scorer, phrase_len=5)
            phrase_score_post, paragraph_score_post = compute_external_rouge_score(paragraphs=post_think_paragraphs, scorer=scorer)
            
            # Weighted combination of scores
            if phrase_score == 0.0 and paragraph_score == 0.0:
                phrase_score = (phrase_score_post + internal_phrase_score_post) / 2
                paragraph_score = paragraph_score_post
            else:
                phrase_score_post = (phrase_score_post + internal_phrase_score_post) / 2
                phrase_score = (phrase_score + phrase_score_post) / 2
                paragraph_score = (paragraph_score + paragraph_score_post) / 2

    else:
        ### </think>가 없는 경우 추출
        # Check repetition in the entire text if no </think> sections are found
        pre_tail_text = solution_str[-tail_token_num:]
        
        pre_paragraphs = [p.strip() for p in pre_tail_text.split('\n\n') if p.strip()]

        if len(pre_paragraphs) >= 2:
            # 마지막 문단에서 내부 phrase 체크 한 번 더 하기
            internal_phrase_score_pre = compute_internal_rouge_score(paragraph=pre_paragraphs[-1], scorer=scorer, phrase_len=5)
            phrase_score_pre, paragraph_score_pre = compute_external_rouge_score(paragraphs=pre_paragraphs, scorer=scorer)
            
            # Average scores across all scenarios
            if phrase_score == 0.0 and paragraph_score == 0.0:
                phrase_score = (phrase_score_pre + internal_phrase_score_pre) / 2
                paragraph_score = paragraph_score_pre
            else:
                phrase_score_pre = (phrase_score_pre + internal_phrase_score_pre) / 2
                phrase_score = (phrase_score + phrase_score_pre) / 2
                paragraph_score = (paragraph_score + paragraph_score_pre) / 2

    return char_score, phrase_score, paragraph_score