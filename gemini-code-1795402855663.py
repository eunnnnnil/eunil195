import os
import random
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- 코랩 한글 폰트 자동 설치 및 설정 (네모 깨짐 방지) ---
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(font_path):
    subprocess.run(["apt-get", "-qq", "update"], check=False)
    subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-nanum"], check=False)

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

DAYS_KR = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
SUBJECT_COLORS = [
    "#4A90E2", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899",
    "#06B6D4", "#EF4444", "#84CC16", "#6366F1", "#14B8A6",
]

task_list = [
    {"name": "수학 기출문제 풀이", "base_hours": 4.0, "difficulty": 0.9, "urgency": 0.8},
    {"name": "AI 수행평가 보고서", "base_hours": 3.5, "difficulty": 0.7, "urgency": 0.9},
    {"name": "영어 단어 및 독해", "base_hours": 3.0, "difficulty": 0.4, "urgency": 0.5},
]


class ColabScheduleOptimizer:
    def __init__(self, tasks, weekday_max, weekend_max, weekend_focus, fatigue, speed, completion_rate, buffer_ratio=0.15):
        self.tasks = tasks
        self.profile = {
            "weekday_max_hours": weekday_max,
            "weekend_max_hours": weekend_max,
            "weekend_focus": weekend_focus,
            "fatigue_level": fatigue,
            "learning_speed": speed,
            "completion_rate": completion_rate,
            "buffer_ratio": buffer_ratio,
        }
        self.days = 7
        self.weights = {
            "fatigue_penalty": 2.5,
            "cramming_penalty": 4.5,
            "urgency_bonus": 0.4,
            "balance_penalty": 2.5,
        }

    def get_daily_max(self, day):
        return self.profile["weekend_max_hours"] if day >= 5 else self.profile["weekday_max_hours"]

    def get_effective_daily_limit(self, day):
        return round(self.get_daily_max(day) * (1.0 - self.profile["buffer_ratio"]), 1)

    def estimate_actual_hours(self, task):
        adjusted = task["base_hours"] / self.profile["learning_speed"]
        buffer_factor = 1.0 + (1.0 - self.profile["completion_rate"]) * 0.4
        return round(adjusted * buffer_factor, 1)

    def create_random_schedule(self, start_day=0):
        schedule = {day: [] for day in range(self.days)}
        day_indices = list(range(start_day, self.days))
        day_weights = [
            (1.0 + self.profile["weekend_focus"] * 2.5) if d >= 5 else 1.0
            for d in day_indices
        ]
        for idx, task in enumerate(self.tasks):
            needed_blocks = max(1, int(self.estimate_actual_hours(task) * 2))
            for _ in range(needed_blocks):
                chosen_day = random.choices(day_indices, weights=day_weights, k=1)[0]
                schedule[chosen_day].append(idx)
        return schedule

    def fitness(self, schedule, start_day=0):
        score = 100.0
        fatigue = self.profile["fatigue_level"]
        weekend_focus = self.profile["weekend_focus"]

        weekday_hours_list = []
        weekend_hours_list = []

        for day in range(start_day, self.days):
            daily_blocks = schedule[day]
            daily_hours = len(daily_blocks) * 0.5
            effective_limit = self.get_effective_daily_limit(day)

            if day >= 5:
                weekend_hours_list.append(daily_hours)
            else:
                weekday_hours_list.append(daily_hours)

            if daily_hours > effective_limit:
                score -= (daily_hours - effective_limit) * self.weights["cramming_penalty"] * 6

            daily_difficulty = sum(self.tasks[t_idx]["difficulty"] for t_idx in daily_blocks)
            if fatigue > 0.5 and daily_difficulty > 2.2:
                score -= daily_difficulty * self.weights["fatigue_penalty"]

            if day >= 5:
                score += daily_hours * weekend_focus * 4.5
            else:
                for t_idx in daily_blocks:
                    score += self.tasks[t_idx]["urgency"] * (5 - day) * self.weights["urgency_bonus"] * 0.08

            fatigue = min(1.0, fatigue + 0.15) if daily_hours > effective_limit * 0.9 else max(0.1, fatigue - 0.05)

        if len(weekday_hours_list) > 1:
            avg_wd = sum(weekday_hours_list) / len(weekday_hours_list)
            var_wd = sum((h - avg_wd) ** 2 for h in weekday_hours_list) / len(weekday_hours_list)
            score -= var_wd * self.weights["balance_penalty"] * 3.5

        if len(weekend_hours_list) > 1:
            avg_we = sum(weekend_hours_list) / len(weekend_hours_list)
            var_we = sum((h - avg_we) ** 2 for h in weekend_hours_list) / len(weekend_hours_list)
            score -= var_we * self.weights["balance_penalty"] * 4.0

        return score

    def crossover_and_mutate(self, p1, p2, start_day=0):
        child = {day: [] for day in range(self.days)}
        day_indices = list(range(start_day, self.days))
        day_weights = [
            (1.0 + self.profile["weekend_focus"] * 2.5) if d >= 5 else 1.0
            for d in day_indices
        ]
        for t_idx in range(len(self.tasks)):
            source = p1 if random.random() < 0.5 else p2
            for day in range(start_day, self.days):
                count = source[day].count(t_idx)
                if count > 0 and random.random() < 0.15:
                    count -= 1
                    chosen_day = random.choices(day_indices, weights=day_weights, k=1)[0]
                    child[chosen_day].append(t_idx)
                child[day].extend([t_idx] * count)
        return child

    def optimize(self, generations=100, pop_size=50, start_day=0):
        population = [self.create_random_schedule(start_day) for _ in range(pop_size)]
        for _ in range(generations):
            population.sort(key=lambda s: self.fitness(s, start_day), reverse=True)
            next_gen = population[:10]
            while len(next_gen) < pop_size:
                p1, p2 = random.sample(population[:20], 2)
                next_gen.append(self.crossover_and_mutate(p1, p2, start_day))
            population = next_gen
        return population[0]


# --- 1. 과목 직접 입력 위젯 ---
style = {'description_width': 'initial'}
w_task_name = widgets.Text(placeholder='예: 국어 비문학 독해', description='과목/할 일:', style=style)
w_task_hours = widgets.FloatSlider(value=2.5, min=0.5, max=15.0, step=0.5, description='예상 소요시간(h):', style=style)
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
    w_task_name.value = ""
    render_task_list()


def on_clear_tasks(b):
    task_list.clear()
    render_task_list()


btn_add.on_click(on_add_task)
btn_clear.on_click(on_clear_tasks)

# --- 2. 평일/주말 가용 시간 및 주말 집중도 설정 위젯 ---
w_weekday_max = widgets.FloatSlider(value=2.5, min=0.5, max=8.0, step=0.5, description='평일(월~금) 하루 최대(h):', style=style)
w_weekend_max = widgets.FloatSlider(value=6.0, min=2.0, max=12.0, step=0.5, description='주말(토·일) 하루 최대(h):', style=style)
w_weekend_focus = widgets.FloatSlider(value=0.8, min=0.0, max=1.0, step=0.1, description='주말 집중 배정 강도:', style=style)
w_buffer = widgets.FloatSlider(value=0.15, min=0.0, max=0.4, step=0.05, description='돌발변수 예비버퍼 비율:', style=style)
w_fatigue = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, step=0.1, description='현재 누적 피로도:', style=style)
w_speed = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1, description='학습 속도 배율:', style=style)
w_comp = widgets.FloatSlider(value=0.8, min=0.3, max=1.0, step=0.05, description='과거 계획 이행률:', style=style)
btn_run = widgets.Button(description='주말 중심 최적 일정 생성하기', button_style='primary')

# --- 3. 변수 사유 선택 및 직접 입력 위젯 ---
EVENT_PRESETS = {
    "학원 / 보충수업": {"default_hours": 2.0, "fatigue_delta": +0.15},
    "학교 행사 / 동아리 / 야자": {"default_hours": 1.5, "fatigue_delta": +0.10},
    "컨디션 난조 / 병원 / 휴식": {"default_hours": 2.0, "fatigue_delta": -0.15},
    "가족 행사 / 개인 약속": {"default_hours": 2.5, "fatigue_delta": +0.05},
    "기타 (직접 입력)": {"default_hours": 1.5, "fatigue_delta": +0.05},
}

w_event_day = widgets.Dropdown(options=[(DAYS_KR[i], i) for i in range(6)], value=0, description='변수 발생 요일:', style=style)
w_event_type = widgets.Dropdown(options=list(EVENT_PRESETS.keys()), value="학원 / 보충수업", description='변수 사유 선택:', style=style)
w_event_custom = widgets.Text(placeholder='예: 수학 단과 특강, 치과 예약 등 (선택사항)', description='상세 사유 직접 입력:', style=style)
w_event_loss = widgets.FloatSlider(value=2.0, min=0.5, max=6.0, step=0.5, description='차감할 공부 시간(h):', style=style)
btn_replan = widgets.Button(description='⚡ 변수 사유 반영 & 남은 요일 재분배', button_style='danger')

# 사유 선택 시 기본 추천 시간 자동 세팅
def on_event_type_change(change):
    preset = EVENT_PRESETS[change["new"]]
    w_event_loss.value = preset["default_hours"]

w_event_type.observe(on_event_type_change, names="value")

out_result = widgets.Output()
current_schedule = {}


def render_schedule_and_chart(schedule, weekday_max, weekend_max, buffer_ratio, note=""):
    with out_result:
        clear_output()
        if note:
            print(f"📢 {note}\n")
        print(f"=== 맞춤형 주간 학습 시간표 (평일 최대 {weekday_max}h / 주말 최대 {weekend_max}h) ===")

        daily_totals = [0.0] * 7
        daily_max_list = [weekend_max if d >= 5 else weekday_max for d in range(7)]
        subject_daily_hours = {t_idx: [0.0] * 7 for t_idx in range(len(task_list))}

        for d in range(7):
            blocks = schedule[d]
            total_h = len(blocks) * 0.5
            daily_totals[d] = total_h
            summary = {}
            for t_idx in blocks:
                subject_daily_hours[t_idx][d] += 0.5
                name = task_list[t_idx]["name"]
                summary[name] = summary.get(name, 0) + 0.5
            buffer_left = max(0.0, round(daily_max_list[d] - total_h, 1))
            tag = "[주말 집중]" if d >= 5 else "[평일]"
            text = ", ".join([f"{k}({v}시간)" for k, v in summary.items()]) or "휴식"
            print(f"• {DAYS_KR[d]} {tag} (학습 {total_h:.1f}h / 예비버퍼 {buffer_left:.1f}h): {text}")

        plt.figure(figsize=(9.5, 4.5))
        bottoms = [0.0] * 7

        for t_idx, task in enumerate(task_list):
            hours_per_day = subject_daily_hours[t_idx]
            color = SUBJECT_COLORS[t_idx % len(SUBJECT_COLORS)]
            plt.bar(
                DAYS_KR, hours_per_day, bottom=bottoms, color=color,
                edgecolor="white", linewidth=0.8, width=0.52, label=task["name"]
            )
            for d in range(7):
                if hours_per_day[d] >= 0.5:
                    plt.text(
                        d, bottoms[d] + hours_per_day[d] / 2, f"{hours_per_day[d]:.1f}h",
                        ha="center", va="center", color="white", fontsize=8.5, fontweight="bold"
                    )
            bottoms = [bottoms[d] + hours_per_day[d] for d in range(7)]

        buffer_heights = [max(0.0, daily_max_list[d] - daily_totals[d]) for d in range(7)]
        plt.bar(
            DAYS_KR, buffer_heights, bottom=bottoms, color="#E2E8F0", alpha=0.45,
            edgecolor="#94A3B8", linestyle="--", hatch="//", width=0.52, label="돌발변수 대비 예비 시간(버퍼)"
        )

        plt.step(range(7), daily_max_list, where="mid", color="#E74C3C", linestyle="--", linewidth=1.4, label="요일별 최대 가용 한도")

        for d, total_h in enumerate(daily_totals):
            top_y = max(total_h, daily_max_list[d])
            plt.text(d, top_y + 0.15, f"학습 {total_h:.1f}h", ha="center", fontsize=9, fontweight="bold", color="#1E293B")

        plt.ylim(0, max(max(daily_totals) + 1.8, max(daily_max_list) + 1.5))
        plt.ylabel("학습 시간 (시간)")
        plt.title("주말(토·일) 집중형 주간 학습 시간 배분 결과")
        plt.legend(loc="upper left", frameon=True, fontsize=8.5, ncol=2)
        plt.tight_layout()
        plt.show()


def on_click_run(b):
    global current_schedule
    if not task_list:
        with out_result:
            clear_output()
            print("⚠️ 먼저 최소 1개 이상의 과목을 추가해 주세요.")
        return

    opt = ColabScheduleOptimizer(
        task_list, w_weekday_max.value, w_weekend_max.value, w_weekend_focus.value,
        w_fatigue.value, w_speed.value, w_comp.value, w_buffer.value
    )
    current_schedule = opt.optimize()
    render_schedule_and_chart(current_schedule, w_weekday_max.value, w_weekend_max.value, w_buffer.value)


def on_click_replan(b):
    global current_schedule
    if not current_schedule:
        with out_result:
            clear_output()
            print("⚠️ 먼저 [주말 중심 최적 일정 생성하기]를 눌러 기본 시간표를 생성해 주세요.")
        return

    ev_day = w_event_day.value
    ev_type = w_event_type.value
    custom_txt = w_event_custom.value.strip()
    reason_label = f"{ev_type} ({custom_txt})" if custom_txt else ev_type

    # 선택한 변수 사유에 따른 피로도 자동 보정
    fatigue_delta = EVENT_PRESETS[ev_type]["fatigue_delta"]
    adjusted_fatigue = round(min(1.0, max(0.0, w_fatigue.value + fatigue_delta)), 2)
    w_fatigue.value = adjusted_fatigue

    loss_blocks = int(w_event_loss.value * 2)
    removed_blocks = []
    for _ in range(min(loss_blocks, len(current_schedule[ev_day]))):
        removed_blocks.append(current_schedule[ev_day].pop())

    if not removed_blocks:
        render_schedule_and_chart(
            current_schedule, w_weekday_max.value, w_weekend_max.value, w_buffer.value,
            note=f"[{reason_label}] {DAYS_KR[ev_day]}에는 원래 배정된 자습이 없어 기존 일정을 유지합니다. (누적 피로도 → {adjusted_fatigue})"
        )
        return

    opt = ColabScheduleOptimizer(
        task_list, w_weekday_max.value, w_weekend_max.value, w_weekend_focus.value,
        adjusted_fatigue, w_speed.value, w_comp.value, 0.0
    )
    best_replan = None
    best_score = -1e9

    day_indices = list(range(ev_day + 1, 7))
    day_weights = [(1.0 + w_weekend_focus.value * 2.5) if d >= 5 else 1.0 for d in day_indices]

    for _ in range(60):
        candidate = {d: list(current_schedule[d]) for d in range(7)}
        for t_idx in removed_blocks:
            target_day = random.choices(day_indices, weights=day_weights, k=1)[0]
            candidate[target_day].append(t_idx)
        sc = opt.fitness(candidate, start_day=ev_day + 1)
        if sc > best_score:
            best_score = sc
            best_replan = candidate

    current_schedule = best_replan
    w_event_custom.value = ""
    render_schedule_and_chart(
        current_schedule, w_weekday_max.value, w_weekend_max.value, w_buffer.value,
        note=f"⚡ [{reason_label}] 사유로 {DAYS_KR[ev_day]} 자습 {len(removed_blocks)*0.5:.1f}h 차감 (피로도 {adjusted_fatigue} 반영) → 주말 및 남은 요일로 재분배 완료!"
    )


btn_run.on_click(on_click_run)
btn_replan.on_click(on_click_replan)

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
    widgets.HTML("<hr><h3>2단계: 평일/주말 가용 시간 및 주말 집중도 설정</h3>"),
    w_weekday_max,
    w_weekend_max,
    w_weekend_focus,
    w_buffer,
    w_fatigue,
    w_speed,
    w_comp,
    btn_run,
    widgets.HTML("<hr><h3>3단계: 돌발 변수(학원·일정 등) 발생 시 사유 선택 및 재조정</h3>"),
    widgets.HBox([w_event_day, w_event_type]),
    w_event_custom,
    w_event_loss,
    btn_replan,
    out_result
]))