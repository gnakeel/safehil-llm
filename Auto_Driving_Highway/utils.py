# utils.py
def extract_decision(response_content):
    try:
        import re
        pattern = r'"decision":\s*{\s*"([^"]+)"\s*}'
        match = re.search(pattern, response_content)
        if match:
            raw_decision = match.group(1).upper().strip()
            
            # 매핑 사전
            decision_map = {
                "ACCELERATE": "FASTER",
                "SPEED UP": "FASTER",
                "GO FASTER": "FASTER",
                "DECELERATE": "SLOWER",
                "SLOW DOWN": "SLOWER",
                "BRAKE": "SLOWER",
                "STAY": "IDLE",
                "MAINTAIN": "IDLE",
                "KEEP": "IDLE",
                "LEFT": "LANE_LEFT",
                "CHANGE LEFT": "LANE_LEFT",
                "RIGHT": "LANE_RIGHT",
                "CHANGE RIGHT": "LANE_RIGHT"
            }
            
            # 매핑 시도
            if raw_decision in {'FASTER', 'SLOWER', 'LEFT', 'IDLE', 'RIGHT'}:
                return raw_decision  # 이미 올바른 형식
            else:
                mapped = decision_map.get(raw_decision)
                print(f"LLM 응답 '{raw_decision}'을(를) '{mapped}'(으)로 매핑")
                return mapped
        
        print(f"결정 패턴을 찾을 수 없음: {response_content}")
        return None
    except Exception as e:
        print(f"결정 추출 중 오류: {e}")
        return None
