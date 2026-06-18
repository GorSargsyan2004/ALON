$BaseUrl = "http://localhost:1234/v1"

Write-Host "GET $BaseUrl/models"
$models = Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get
$models | ConvertTo-Json -Depth 5 | Write-Host

Write-Host "POST $BaseUrl/chat/completions"
$body = @{
  model = ($models.data[0].id)
  messages = @(
    @{ role = "system"; content = "You are a concise assistant." },
    @{ role = "user"; content = "Say hello in one sentence." }
  )
  temperature = 0.6
  max_tokens = 64
} | ConvertTo-Json -Depth 5

$resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -Body $body -ContentType "application/json"
$resp.choices[0].message.content | Write-Host
