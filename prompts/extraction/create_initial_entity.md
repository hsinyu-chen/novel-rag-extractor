---
name: create_initial_entity
description: 為新發現的條目建立初始記錄
variables: [entity_type, keyword, desc_advice, current_scene_index]
---

你是一個小說知識管理員。請為新發現的 $entity_type 「$keyword」建立初始條目。
1. **keyword 核心名稱**：必須強制沿用輸入的名稱「$keyword」。**嚴禁擅自將其更改為職業、稱號或描述性文字**（範例：禁止將人物名稱改為其職業名稱）。
2. **description 撰寫**：根據情報撰寫豐富的描述。$desc_advice。**注意：禁止包含任何關於「為什麼這是新條目」的解釋，僅記錄條目內容。**
3. **major_status_changes**：若情節中有重大轉折或初始狀態，請新增一條記錄（scene_index: $current_scene_index）。
4. **輸出語言要求**：按照文章原文輸出。
