$baseUrl = "http://localhost:1234/v1"
$model = "qwen2.5-3b-instruct"

$schema = @{
  type = "object"
  properties = @{
    intent = @{ type = "string"; enum = @("assist","weather","search","filesystem","executor") }
    confidence = @{ type = "number"; minimum = 0; maximum = 1 }
    need_context = @{
      anyOf = @(
        @{ type = "null" },
        @{
          type = "object"
          properties = @{
            start = @{ type = @("string","null") }
            end = @{ type = @("string","null") }
            reason = @{ type = @("string","null") }
          }
          required = @("start","end","reason")
          additionalProperties = $false
        }
      )
    }
    tool_calls = @{
      type = "array"
      items = @{
        type = "object"
        properties = @{
          name = @{ type = "string" }
          arguments = @{ type = "object" }
        }
        required = @("name","arguments")
        additionalProperties = $false
      }
    }
    user_rewrite = @{ type = "string" }
  }
  required = @("intent","confidence","need_context","tool_calls","user_rewrite")
  additionalProperties = $false
}

$body = @{
  model = $model
  messages = @(
    @{ role = "system"; content = "Return strict JSON only for routing." }
    @{ role = "user"; content = "Alon do you remember what happened yesterday?" }
  )
  temperature = 0.2
  max_tokens = 350
  response_format = @{
    type = "json_schema"
    json_schema = @{
      name = "router_decision"
      schema = $schema
      strict = $true
    }
  }
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Uri "$baseUrl/chat/completions" -Method Post -Body $body -ContentType "application/json"
$resp.choices[0].message.content
