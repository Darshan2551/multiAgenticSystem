import re
import json

def robust_clean_json(raw_text):
    """
    Attempt to extract a single JSON object from raw_text.
    1) Strip surrounding whitespace.
    2) Remove leading/trailing markdown fences (```json etc).
    3) Find the first '{' and the matching closing '}' (simple bracket matching).
    4) Return the substring or raise ValueError if not found.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Input to robust_clean_json must be a string.")
    s = raw_text.strip()

    # remove common fences and leading labels
    # remove fenced code blocks (```json ... ``` or ``` ... ```)
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # remove possible leading lines like "json" or "JSON" or "output:"
    s = re.sub(r'^(json|JSON|output:|Output:)\s*', "", s)

    # find first '{'
    start = s.find('{')
    if start == -1:
        raise ValueError("No JSON object start '{' found in LLM output.")

    # simple bracket matching to find corresponding closing brace
    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Could not find matching '}' for JSON object in LLM output.")

    candidate = s[start:end+1].strip()

    # final sanity: try to parse it
    try:
        parsed = json.loads(candidate)
        return candidate  # return the clean JSON string (or return parsed if you prefer)
    except Exception as e:
        raise ValueError(f"Extracted JSON failed to parse: {e}\nExtracted text:\n{candidate}")

import json
import random

def load_rules():
    with open("quant_design_rules.json", "r") as f:
        return json.load(f)

rules = load_rules()

def generator_prompt():
    # Convert rules to a readable template for LLM
    rules_block = json.dumps(rules, indent=2)

    return f"""
You are the Quant Problem Generator Agent.

Below are the design rules you MUST follow:
{rules_block}

Generate ONE brand new quant question following the required JSON schema.

RULES:
- Use realistic numbers.
- Must belong to any of these categories: {rules['output_schema']['category']}
- Follow one of the templates.
- Ensure hidden_solution matches the correct answer.
- Provide 4 MCQs, unique, one correct.
- Return ONLY JSON. No markdown.

Output MUST be valid JSON and match this structure:

{{
  "id": "string",
  "category": "string",
  "difficulty": "Easy|Medium|Hard",
  "question": "string",
  "variables": {{}},
  "options": {{"A": 0, "B": 0, "C": 0, "D": 0}},
  "correct_option": "A",
  "hidden_solution": "string",
  "units": "string"
}}
"""


def generator_agent(call_llm):
    prompt = generator_prompt()
    raw = call_llm(prompt)

    if raw is None:
        print("Generator returned None")
        return None

    cleaned = raw.strip()

    # Remove common markdown fences quickly
    if cleaned.startswith("```"):
        # if model returned ```json\n{...}\n```, take inner content first
        parts = cleaned.split("```")
        # find first non-empty segment that contains '{'
        inner = next((p for p in parts if '{' in p), cleaned)
        cleaned = inner.strip()

    # Try normal json.loads first (fast path)
    try:
        return json.loads(cleaned)
    except Exception as e1:
        # Defensive: attempt robust extraction
        try:
            jstr = robust_clean_json(raw)
            return json.loads(jstr)
        except Exception as e2:
            print("Generator JSON ERROR:", e2)
            print("LLM OUTPUT (raw):", raw[:1000])  # print first 1000 chars for debug
            return None

def solver_A_prompt(question_json):
    return f"""
You are Solver Agent A.
You must solve the following quantitative word problem using correct algebra and step-by-step reasoning.

IMPORTANT RULES:
- Use ONLY the values given in the problem.
- Build equations clearly.
- Show ALL steps of reasoning.
- Do not skip conversions.
- Do not hallucinate extra information.
- Do NOT modify any numbers in the story.
- Produce ONE numeric final answer.
- Output ONLY valid JSON.

Return exactly this JSON format:

{{
  "equations": ["..."],
  "steps": ["..."],
  "final_answer": 0,
  "confidence": "High|Medium|Low"
}}

Problem to solve:
{json.dumps(question_json, indent=2)}
"""

def solver_agent_A(question_json, call_llm):
    prompt = solver_A_prompt(question_json)
    raw = call_llm(prompt)

    if raw is None:
        print("Solver A: No output from LLM.")
        return None

    text = raw.strip()

    # Remove ```json or ``` wrappers
    if text.startswith("```"):
        parts = text.split("```")
        text = next((p for p in parts if "{" in p), text).strip()

    # Remove accidental prefixes like: json, JSON, Output:
    for prefix in ["json", "JSON", "Json", "output:", "Output:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Extract pure JSON block using bracket matching
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        print("Solver A JSON ERROR: No JSON object found.")
        print("LLM OUTPUT:", raw)
        return None

    json_str = text[start:end+1]

    try:
        data = json.loads(json_str)
        return data
    except Exception as e:
        print("Solver A JSON PARSE ERROR:", e)
        print("Extracted JSON:", json_str)
        return None
    
def solver_B_prompt(question_json):
    return f"""
You are Solver Agent B.

Your job is to solve the following quantitative word problem WITHOUT using algebra.
Instead use:
- reverse checking,
- brute-force substitution,
- option elimination,
- logic-based reasoning.

RULES:
- Check each MCQ option to see which one satisfies the problem.
- Do NOT modify story values.
- Do NOT add new numbers.
- No algebraic manipulation like Solver A.
- Only check by plugging values back.

Return ONLY valid JSON in this format:

{{
  "checked_options": {{}},
  "final_answer": 0,
  "reasoning": ["..."],
  "confidence": "High|Medium|Low"
}}

Problem:
{json.dumps(question_json, indent=2)}
"""

def solver_agent_B(question_json, call_llm):
    prompt = solver_B_prompt(question_json)
    raw = call_llm(prompt)

    if raw is None:
        print("Solver B: No output from LLM.")
        return None

    text = raw.strip()

    # Remove code fences ```json ... ```
    if text.startswith("```"):
        parts = text.split("```")
        text = next((p for p in parts if "{" in p), text).strip()

    # Strip leading prefixes like: json, JSON, output:
    for prefix in ["json", "JSON", "Json", "Output:", "output:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Extract JSON using bracket matching
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        print("Solver B JSON ERROR: No JSON object found.")
        print("LLM OUTPUT:", raw)
        return None

    json_str = text[start:end+1]

    try:
        data = json.loads(json_str)
        return data
    except Exception as e:
        print("Solver B JSON PARSE ERROR:", e)
        print("Extracted JSON:", json_str)
        return None
    
def validate_question(question, solverA, solverB):
    issues = []

    # 1. Check both solvers returned data
    if solverA is None:
        issues.append("Solver A returned no data.")
    if solverB is None:
        issues.append("Solver B returned no data.")

    if issues:
        return False, issues

    # 2. Compare answers numerically
    try:
        a = float(solverA["final_answer"])
        b = float(solverB["final_answer"])
        if abs(a - b) > 1e-6:
            issues.append(f"Solver mismatch: A={a}, B={b}")
    except:
        issues.append("Could not parse answers from solvers.")

    # 3. Check MCQ contains the answer
    correct_option = question["correct_option"]
    correct_value = question["options"][correct_option]

    if abs(float(correct_value) - a) > 1e-6:
        issues.append(f"Correct option value ({correct_value}) does NOT match solver answer ({a}).")

    # 4. Check for impossible values
    # Example: negative time, negative speed
    if "variables" in question:
        for k, v in question["variables"].items():
            if isinstance(v, (int, float)) and v < 0:
                issues.append(f"Invalid variable: {k} is negative.")

    # 5. Check hidden_solution exists
    if "hidden_solution" not in question or not question["hidden_solution"]:
        issues.append("Missing hidden_solution.")

    # 6. Check required fields exist
    required_fields = ["id", "category", "question", "options", "correct_option", "units"]
    for field in required_fields:
        if field not in question:
            issues.append(f"Missing field in question: {field}")

    # Decide validity
    is_valid = (len(issues) == 0)
    return is_valid, issues

def validator_agent(question, solverA_output, solverB_output):
    is_valid, issues = validate_question(question, solverA_output, solverB_output)

    if is_valid:
        return {
            "status": "VALID",
            "issues": [],
            "final_answer": solverA_output["final_answer"]
        }
    else:
        return {
            "status": "INVALID",
            "issues": issues,
            "regenerate": True
        }
    
import requests, json

API_KEY = "YOUR API KEY HERE"

response = requests.get(
    f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"
)

print(json.dumps(response.json(), indent=2))


def orchestrator(call_llm, max_questions=5, max_attempts=50):
    validated = []
    stats = {
        "total_generated": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "attempts": 0
    }

    for _ in range(max_attempts):
        stats["attempts"] += 1
        print(f"\n=== Attempt {stats['attempts']} ===")

        # Generate question
        q = generator_agent(call_llm)
        if q is None:
            print("❌ Generator returned invalid JSON.")
            stats["invalid_count"] += 1
            continue

        stats["total_generated"] += 1

        # Solver A
        a = solver_agent_A(q, call_llm)
        if a is None:
            print("❌ Solver A failed.")
            stats["invalid_count"] += 1
            continue

        # Solver B
        b = solver_agent_B(q, call_llm)
        if b is None:
            print("❌ Solver B failed.")
            stats["invalid_count"] += 1
            continue

        # VALIDATION (FIXED)
        valid, reason = validate_question(q, a, b)
        if not valid:
            print(f"❌ Validator failed: {reason}")
            stats["invalid_count"] += 1
            continue

        # SUCCESS
        print("✅ VALID QUESTION ADDED!")
        validated.append({"question": q, "solverA": a, "solverB": b})
        stats["valid_count"] += 1

        if stats["valid_count"] >= max_questions:
            print("\n🎉 ALL VALID QUESTIONS GENERATED!")
            break

    return validated, stats

import requests
import json

API_KEY = r'YOUR API KEY HERE'  # your correct cloud key

def real_llm(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    # --- CASE 1: Normal candidate response ---
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        pass

    # --- CASE 2: Some Gemini responses return in a 'text' field ---
    try:
        return data["text"]
    except:
        pass

    # --- CASE 3: parts array (alternate structure) ---
    try:
        return data["candidates"][0]["content"]["parts"][0].get("text", "")
    except:
        pass

    # --- CASE 4: Gemini safety filters triggered ---
    if "promptFeedback" in data:
        print("⚠️ Safety filter triggered:", json.dumps(data["promptFeedback"], indent=2))
        return None

    # --- CASE 5: error response ---
    if "error" in data:
        print("❌ Gemini API Error:", json.dumps(data["error"], indent=2))
        return None

    # --- FALLBACK DEBUG: print full response ---
    print("❌ FULL GEMINI RESPONSE (unrecognized format):")
    print(json.dumps(data, indent=2))

    raise KeyError("Gemini did not return any compatible text output.")

def generate_interactive_quiz(validated_questions):
    html = """
<html>
<head>
<title>Quant Quiz</title>
<style>
body { font-family: Arial; margin: 40px; }
.question { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
.options { margin-left: 20px; }
.correctAns { color: green; font-weight: bold; }
.wrongAns { color: red; font-weight: bold; }
.hidden { display: none; }
button { padding: 10px 20px; margin-top: 20px; font-size: 16px; }
</style>
</head>
<body>

<h1>Quantitative MCQ Quiz</h1>
<form id="quizForm">
"""
    for idx, item in enumerate(validated_questions, start=1):
        q = item["question"]
        correct = q["correct_option"]
        correct_value = q["options"][correct]

        html += f"""
<div class='question'>
<h3>Q{idx}. {q['question']}</h3>
<div class='options'>
<label><input type='radio' name='q{idx}' value='A'> A. {q['options']['A']}</label><br>
<label><input type='radio' name='q{idx}' value='B'> B. {q['options']['B']}</label><br>
<label><input type='radio' name='q{idx}' value='C'> C. {q['options']['C']}</label><br>
<label><input type='radio' name='q{idx}' value='D'> D. {q['options']['D']}</label><br>
</div>

<p id='ans{idx}' class='hidden'>
Correct Answer: <b>{correct} ({correct_value})</b>
</p>

</div>
"""
    # JS Logic
    html += """
<button type="button" onclick="submitQuiz()">Submit Quiz</button>
</form>

<h2 id="score"></h2>

<script>
function submitQuiz() {
    let score = 0;
"""

    for idx, item in enumerate(validated_questions, start=1):
        correct = item["question"]["correct_option"]

        html += f"""
    var selected{idx} = document.querySelector("input[name='q{idx}']:checked");
    var ansBox{idx} = document.getElementById('ans{idx}');

    if (selected{idx}) {{
        if (selected{idx}.value == "{correct}") {{
            score++;
            ansBox{idx}.className = 'correctAns';
        }} else {{
            ansBox{idx}.className = 'wrongAns';
        }}
    }} else {{
        ansBox{idx}.className = 'wrongAns';
    }}

    ansBox{idx}.style.display = 'block';
"""

    total = len(validated_questions)
    html += f"""
    document.getElementById("score").innerHTML =
        "Your Score: " + score + " / {total}";
        
    
</script>


</body>
</html>
"""
    return html


def save_interactive_quiz(validated_questions, filename="quiz.html"):
    html = generate_interactive_quiz(validated_questions)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print("Interactive quiz saved as", filename)



def generate_accuracy_report(stats):
    total = stats["total_generated"]
    valid = stats["valid_count"]
    invalid = stats["invalid_count"]

    acceptance_rate = (valid / total * 100) if total > 0 else 0

    report = {
        "total_generated": total,
        "valid_questions": valid,
        "invalid_questions": invalid,
        "acceptance_rate_percent": round(acceptance_rate, 2),
        "total_attempts": stats["attempts"]
    }

    return report

def save_accuracy_report(report, filename="accuracy_report.json"):
    import json
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Accuracy report saved as {filename}")


