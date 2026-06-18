$baseUrl = "http://localhost:1234/v1"
$model = "qwen2.5-7b-instruct-uncensored"

$system = "You are Alon. Keep answers short and natural. Put URLs in parentheses."
$user = @"
Relevant memory transcript:
<2026-02-10 10:00:00> Gor: What's the weather today?

Tool results (JSON):
{ "tools": [ { "tool": "weather.get_forecast", "ok": true, "result": { "location": "Yerevan", "when": "today", "t_min": 0, "t_max": 7, "precip_prob_max": 20, "wind_max": 9 } } ] }

User request: What's the weather today?
"@

$body = @{
  model = $model
  messages = @(
    @{ role = "system"; content = $system }
    @{ role = "user"; content = $user }
  )
  temperature = 0.6
  max_tokens = 200
} | ConvertTo-Json -Depth 6

$resp = Invoke-RestMethod -Uri "$baseUrl/chat/completions" -Method Post -Body $body -ContentType "application/json"
$resp.choices[0].message.content
