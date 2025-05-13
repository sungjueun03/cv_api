import json
import pandas as pd
import numpy as np
import os

# 현재 recommend_products.py 파일 위치 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Total_DB.csv")
products = pd.read_csv(csv_path, encoding='cp949')

# 고민 키워드
concern_keywords = {
    '모공': ['모공', '피지','노폐물','피부결','각질'],
    '주름': ['주름', '탄력','영양공급','피부활력','피부재생','나이트','아이'],
    '수분': ['수분', '보습'],
    '색소침착': ['미백', '브라이트닝', '비타민', '피부톤', '투명','트러블케어','피부재생','피부보호','스팟','저자극','진정']
}

# ✅ 함수 정의
def recommend_from_json(raw_json):
    decoded = json.loads(raw_json)
    regions = decoded["regions"]

    # 평균 계산
    moisture_avg = np.mean([
        regions.get('이마', {}).get('수분', 0),
        regions.get('왼쪽 볼', {}).get('수분', 0),
        regions.get('오른쪽 볼', {}).get('수분', 0),
        regions.get('턱', {}).get('수분', 0)
    ])
    elasticity_avg = np.mean([
        regions.get('이마', {}).get('탄력', 0),
        regions.get('왼쪽 볼', {}).get('탄력', 0),
        regions.get('오른쪽 볼', {}).get('탄력', 0),
        regions.get('턱', {}).get('탄력', 0)
    ])
    pore_avg = np.mean([
        regions.get('왼쪽 볼', {}).get('모공 개수', 0),
        regions.get('오른쪽 볼', {}).get('모공 개수', 0)
    ])
    pigment_avg = regions.get('전체', {}).get('색소침착 개수', 0)

    # 고민 우선순위
    concern_scores = {
        '모공': (pore_avg - 500) / 500 if pore_avg >= 500 else 0,
        '주름': (50 - elasticity_avg) / 50 if elasticity_avg <= 50 else 0,
        '수분': (55 - moisture_avg) / 55 if moisture_avg <= 55 else 0,
        '색소침착': (pigment_avg - 130) / 130 if pigment_avg >= 130 else 0,
    }
    user_concerns = [k for k, v in sorted(concern_scores.items(), key=lambda x: x[1], reverse=True) if v > 0]

    def score_product(row):
        tags = str(row.get('태그', ''))
        score = 0
        weights = [3, 2, 1]
        for idx, concern in enumerate(user_concerns):
            weight = weights[idx] if idx < len(weights) else 1
            for keyword in concern_keywords.get(concern, []):
                if keyword in tags:
                    score += weight
        return score

    products['score'] = products.apply(score_product, axis=1)
    recommended = products[products['score'] > 0].dropna(subset=[
        '브랜드', '제품명', '용량/가격', '별점', '이미지', '제품링크'
    ]).sort_values(by='score', ascending=False).head(5)

    return recommended[['브랜드', '제품명', '용량/가격', '별점', '이미지', '제품링크']]
