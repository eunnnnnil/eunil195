import random
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

DAYS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAYS_KR = ["월요일", "화요일", "수요일", "목요일", "금요일"]

# 초기 기본 과목 목록 (실행 후 화면에서 자유롭게 추가/초기화 가능)
task_list = [
    {"name": "수학 기출문제 풀이", "base_hours": 4.0, "difficulty": 0.9, "urgency": 0.8},
    {"name": "AI 수행평가 보고서", "base_hours": 3.5, "difficulty": 0.7, "urgency": 0.9},
]

class ColabScheduleOptimizer:
    def __init__(self, tasks, daily_max, fatigue, speed, completion_rate):
        self.tasks = tasks
        self.profile = {
            "daily_max_hours": daily_max,
            "fatigue_level": fatigue,
            "learning_speed": speed,
            "completion_rate": completion_rate,
        }
        self.days = 5
        self.weights = {"fatigue_penalty": 2.0, "cramming_penalty": 3.0, "urgency_bonus": 1.5}

    def estimate_actual_hours(self, task):
        adjusted = task["base_hours"] / self.profile["learning_speed"]
        buffer_factor = 1.0 + (1.0 - self.profile["completion_rate"]) * 0.5
        return round(adjusted * buffer_factor, 1)

    def create_random_schedule(self):
        schedule = {day: [] for day in range(self.days)}
        for idx, task in enumerate(self.tasks):
            needed_blocks = max(1, int(self.estimate_actual_hours(task) * 2))
            for _ in range(needed_blocks):
                schedule[random.randint(0, self.days - 1)].append(idx)
        return schedule

    def fitness(self, schedule):
        score = 100.0
        fatigue = self.profile["fatigue_level"]
        for day in range(self.days):
            daily_blocks = schedule[day]
            daily_hours = len(daily_blocks) * 0.5

            if daily_hours > self.profile["daily_max_hours"]:
                score -= (daily_hours - self.profile["daily_max_hours"]) * self.weights["cramming_penalty"] * 5

            daily_difficulty = sum(self.tasks[t_idx]["difficulty"] for t_idx in daily_blocks)
            if fatigue > 0.5 and daily_difficulty > 2.5:
                score -= daily_difficulty * self.weights["fatigue_penalty"]

            for t_idx in daily_blocks:
                score += self.tasks[t_idx]["urgency"] * (self.days - day) * self.weights["urgency_bonus"] * 0.2

            fatigue = min(1.0, fatigue + 0.15) if daily_hours > self.profile["daily_max_hours"] * 0.9 else max(0.1, fatigue - 0.05)
        return score

    def crossover_and_mutate(self, p1, p2):
        child = {day: [] for day in range(self.days)}
        for t_idx in range(len(self.tasks)):
            source = p1 if random.random() < 0.5 else p2
            for day in range(self.days):
                count = source[day].count(t_idx)
                if count > 0 and random.random() < 0.1:
                    count -= 1
                    child[random.randint(0, self.days - 1)].append(t_idx)
                child[day].extend([t_idx] * count)
        return child

    def optimize(self, generations=80, pop_size=40):
        population = [self.create_random_schedule() for _ in range(pop_size)]
        for _ in range(generations):
            population.sort(key=self.fitness, reverse=True)
            next_gen = population[:10]
            while len(next_gen) < pop_size:
                p1, p2 = random.sample(population[:20], 2)
                next_gen.append(self.crossover_and_mutate(p1, p2))
            population = next_gen
        return population[0]

# --- 1. 과목 직접 입력 위젯 ---
style = {'description_width': 'initial'}
w_task_name = widgets.Text(placeholder='예: 국어 비문학 독해', description='과목/할 일:', style=style)
w_task_hours = widgets.FloatSlider(value=2.5, min=0.5, max=10.0, step=0.5, description='예상 소요시간(h):', style=style)
w_task_diff = widgets.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.1, description='난이도(0.1~1.0):', style=style)
w_task_urg = widgets.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.1, description='긴급도(0.1~1.0):', style=style)

btn_add = widgets.Button(description='+ 과목 추가', button_style='success')
btn_clear = widgets.Button(description='목록 전체 비우기', button_style='warning')
out_list = widgets.Output()

def render_task_list():
    with out_list:
        clear_output()
        if not task_list:
            print("등록된 과목이 없습니다. 위 입력창에서 과목을 추가해 주세요.")
            return
        print(f"현재 등록된 과목 ({len(task_list)}개):")
        for i, t in enumerate(task_list, 1):
            print(f"  {i}. {t['name']} | 기본 {t['base_hours']}h | 난이도 {t['difficulty']} | 긴급도 {t['urgency']}")

def on_add_task(b):
    name = w_task_name.value.strip()
    if not name:
        with out_list:
            print("⚠️ 과목 이름을 입력해 주세요!")
        return
    task_list.append({
        "name": name,
        "base_hours": w_task_hours.value,
        "difficulty": w_task_diff.value,
        "urgency": w_task_urg.value
    })
    w_task_name.value = ""  # 입력 후 칸 비우기
    render_task_list()

def on_clear_tasks(b):
    task_list.clear()
    render_task_list()

btn_add.on_click(on_add_task)
btn_clear.on_click(on_clear_tasks)

# --- 2. 개인 상태 설정 및 최적화 실행 위젯 ---
w_max_h = widgets.FloatSlider(value=4.0, min=1.5, max=10.0, step=0.5, description='하루 최대 공부시간(h):', style=style)
w_fatigue = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, step=0.1, description='현재 누적 피로도:', style=style)
w_speed = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1, description='학습 속도 배율:', style=style)
w_comp = widgets.FloatSlider(value=0.8, min=0.3, max=1.0, step=0.05, description='과거 계획 이행률:', style=style)
btn_run = widgets.Button(description='최적 학습 일정 생성하기', button_style='primary')
out_result = widgets.Output()

def on_click_run(b):
    with out_result:
        clear_output()
        if not task_list:
            print("⚠️ 먼저 최소 1개 이상의 과목을 추가해 주세요.")
            return

        opt = ColabScheduleOptimizer(task_list, w_max_h.value, w_fatigue.value, w_speed.value, w_comp.value)
        best = opt.optimize()

        print("=== 맞춤형 주간 학습 시간표 ===")
        daily_totals = []
        for d in range(5):
            blocks = best[d]
            total_h = len(blocks) * 0.5
            daily_totals.append(total_h)
            summary = {}
            for t_idx in blocks:
                name = task_list[t_idx]["name"]
                summary[name] = summary.get(name, 0) + 0.5
            text = ", ".join([f"{k}({v}h)" for k, v in summary.items()]) or "휴식"
            print(f"• {DAYS_KR[d]} (총 {total_h:.1f}시간): {text}")

        # 막대 그래프 시각화
        plt.figure(figsize=(7, 3.2))
        bars = plt.bar(DAYS_EN, daily_totals, color='#4A90E2', width=0.5)
        plt.axhline(w_max_h.value, color='#E74C3C', linestyle='--', label=f'Max Limit ({w_max_h.value}h)')
        plt.ylim(0, max(max(daily_totals) + 1.5, w_max_h.value + 1.5))
        plt.ylabel("Study Hours")
        plt.title("Optimized Weekly Study Distribution")
        plt.legend()
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval}h", ha='center', fontsize=9)
        plt.show()

btn_run.on_click(on_click_run)

# 전체 UI 화면 출력
render_task_list()
display(widgets.VBox([
    widgets.HTML("<h3>1단계: 내 과목 직접 입력하기</h3>"),
    w_task_name,
    w_task_hours,
    w_task_diff,
    w_task_urg,
    widgets.HBox([btn_add, btn_clear]),
    out_list,
    widgets.HTML("<hr><h3>2단계: 내 컨디션 설정 및 시간표 생성</h3>"),
    w_max_h,
    w_fatigue,
    w_speed,
    w_comp,
    btn_run,
    out_result
]))