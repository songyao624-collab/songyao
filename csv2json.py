import pandas as pd
import json

# Load CSV file
df = pd.read_csv('./fish-2025-dataset.csv')
conversations = []

# Add conversation data
for i in range(len(df)):
    conversations.append({
        "id": f"identity_{i+1}",
        "conversations": [
            {
                "from": "user",
                "value": f"FISH Yes: <|vision_start|>{df.iloc[i]['image_path']}<|vision_end|>"
            },
            {
                "from": "assistant", 
                "value": df.iloc[i]['caption']
            }
        ]
    })

# Save as Json
with open('data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)