GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "location",
    "event",
    "concept",
    "date",
    "technology",
]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to multi-hop question answering, identify only the most important entities and relationships needed to answer factual questions.

-Selection Policy-
Extract a compact, high-signal graph rather than an exhaustive list.
1. Keep at most {max_entities} entities.
2. Keep at most {max_relationships} relationships.
3. Prefer named people, organizations, places, products, works, events, dates, quantities, scientific/common species names, and page-title entities.
4. Prefer entities that are central to the paragraph, appear in factual claims, connect two facts, or are likely to be an answer/supporting fact.
5. Do not extract generic/common words, broad categories, repeated aliases, minor adjectives, or entities mentioned only in passing.
6. If many candidates exist, choose the ones with the clearest factual relationships and ignore the rest.
7. Relationship descriptions must be short factual statements grounded in the text.
8. Relationships are more important than a long entity list. If the output budget is tight, extract fewer entities so that the clearest relationships are still included.

-Steps-
1. Identify the selected entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: concise description of the entity's relevant attributes or role in this text
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
Prefer 1-2 grounded relationships for each central entity when the text supports them. Do not output isolated entities unless they are likely answer entities or page-title anchors.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Interleave relationship records soon after their participating entity records instead of listing every entity first. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["keywords_relationship_extraction"] = """
---Role---

You are a helpful assistant expert in identifying key concepts, specific entities (keywords), and their relationships within the user's query.

---Goal---

Given the query, identify and list the most relevant keywords (encompassing both overarching concepts and specific entities/details). Additionally, articulate the relationships between these keywords, as well as connections to potential answers inferred from the query.

---Instructions---

Output the results in JSON format.
The JSON should have two keys:
"keywords": A list containing the most relevant terms from the query, including both broad concepts/themes and specific entities/details.
"relationships": A list of strings describing the connections between the extracted keywords, and how they relate to the potential meaning or answer implied by the query.
Do not include ``` in the response.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
"keywords": ["International trade", "Global economic stability", "Economic impact", "Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"],
"relationships": [
"International trade influences global economic stability through mechanisms like trade agreements and tariffs.",
"Global economic stability is affected by factors such as currency exchange rates and the balance of imports and exports, which are components of international trade.",
"The economic impact of international trade on global stability is a key area of analysis."
]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
"keywords": ["Environmental consequences", "Deforestation", "Biodiversity", "Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"],
"relationships": [
"Deforestation causes habitat destruction, which is a primary driver of biodiversity loss.",
"Environmental consequences of deforestation include species extinction, increased carbon emissions, and damage to ecosystems like rainforests.",
"Biodiversity is negatively impacted by deforestation through various environmental consequences."
]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
"keywords": ["Education", "Poverty reduction", "Socioeconomic development", "Income inequality", "Primary school enrollment rates", "STEM vocational training", "Microfinance scholarships", "Literacy rates"],
"relationships": [
"Education plays a role in poverty reduction by improving literacy rates and providing skills (e.g., STEM vocational training).",
"Poverty reduction strategies often involve increasing access to education, potentially measured by primary school enrollment rates or aided by microfinance scholarships.",
"Education can contribute to socioeconomic development and potentially help address income inequality, thereby reducing poverty."
]
}}
#############################
Example 4:

Query: "How do convolutional neural networks improve image recognition accuracy?"
################
Output:
{{
"keywords": ["Convolutional neural networks", "Image recognition accuracy", "Feature extraction", "Model accuracy", "Spatial hierarchy learning", "Kernel filters", "Pooling layers", "ReLU activation", "Data augmentation", "Batch normalization"],
"relationships": [
"Convolutional neural networks (CNNs) improve image recognition accuracy through techniques like feature extraction using kernel filters and spatial hierarchy learning.",
"Model accuracy in image recognition is enhanced by CNN components such as pooling layers, ReLU activation, and techniques like data augmentation and batch normalization.",
"Kernel filters extract features, pooling layers reduce dimensionality, ReLU introduces non-linearity, and batch normalization stabilizes training, all contributing to the improved accuracy of CNNs in image recognition."
]
}}
#############################
---Real Data---
Query: {query}
######################
Output:
"""

PROMPTS["naive_rag_response_better"] = """
# Role
You are a helpful AI assistant on question answering.

# Task
Your task is to answer a question referring to a given context, if any.
For answering the Question at the end, you need to first read the context provided, then give your final answer.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.Do not include ``` in your reply.
{{
    "rationale": <a string. Rationale behind your answer.>,
    "answer": <Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>
}}

# Context, if any
{context}

# Question
{query}
"""

PROMPTS["fix_json"] = """
---Role---
You are an expert-level JSON repair tool. Your sole responsibility is to fix improperly formatted JSON strings to make them conform to strict JSON specifications.
---Goal---
Receive a potentially malformed string and correct it into a syntactically valid JSON object or array. During the correction process, you must preserve the intent and content of the original data to the maximum extent possible, and must not modify the semantics of key-value pairs.
---Core Rules---
1.  **Quote Specification**: All keys and string values must be wrapped in double quotes ("). Correct all non-compliant single quotes (') or missing quotes to double quotes.
2.  **Comma Handling**: Handle commas precisely. Remove trailing commas after the last element in an object or array. Add missing commas between elements.
3.  **Bracket Matching**: Ensure all curly braces ({}) and square brackets ([]) are properly paired and closed.
4.  **Data Types**: Ensure strings are wrapped in double quotes, while boolean values (true, false), numbers (e.g., 123, -45.6), and null values should not be wrapped in quotes.
5.  **Escape Characters**: Properly handle escaping within strings. For example, double quotes within strings should be escaped as \", and backslashes should be escaped as \\. Fix all invalid escape sequences.
6.  **Character Standardization**: Replace full-width (Chinese character) punctuation marks, such as `，`, `：`, `"`, `"`, `'`, `'`, `【`, `】`, `｛`, `｝`, with their corresponding half-width (English character) versions.
7.  **Comment Removal**: Remove all JavaScript-style comments, including line comments (starting with //) and block comments (starting with /* and ending with */).
8.  **Root Element**: A valid JSON text must be a single root element (an object or an array). If the input contains multiple root-level objects or values, they should be wrapped in a JSON array.
9.  **Invalid Input Handling**: If the input string is empty, contains only whitespace, or cannot be parsed into any valid JSON structure at all, return an empty JSON object {}.
---Output Format---
-   **Only output** the corrected JSON text.
-   **Do not** include any explanations, descriptions, or text like "Here is the corrected JSON".
-   **Do not** wrap your output with any markers (such as Markdown code blocks).

######################
---
**Example 1**
Input:
{
  'name': 'John Doe',
  "age": 30,
  "isStudent": true,
  "courses": ["Math", "Science",],
}
Output:
{
  "name": "John Doe",
  "age": 30,
  "isStudent": true,
  "courses": [
    "Math",
    "Science"
  ]
}

---
**Example 2**
Input:
｛
  "name"："John Doe"
  "city"："Beijing"，
｝
Output:
{
  "name": "John Doe",
  "city": "Beijing"
}

---
**Example 3**
Input:
{"id": 1} {"id": 2}
Output:
[
  {
    "id": 1
  },
  {
    "id": 2
  }
]

---
**Example 4**
Input:
This is a completely invalid string
Output:
{}
#############################
#############################
-Real Data-
######################
Input:
"""


PROMPTS["eval_answer"] = """
You are a helpful AI assistant good at question answering and answer evaluation.

# Task
Providing a question and its correct answer labels, your task is to analyze whether a given answer is correct or not. An answer is treated as correct if the meaning of any label is expressed in the answer list. Redundant expression in answer is allowed.

# Question
{question}

# Correct Answer Labels
{gold_answer}

# Answer that Requires Judgement
{answer}

Is the answer correct or not? You output should only be "Correct" or "Incorrect".
"""

PROMPTS["eval_docbench_answer"] = """
You are a helpful AI assistant good at document question answering evaluation.

# Task
Given a question, the reference answer, supporting evidence from the original document, and a model prediction, determine whether the prediction is correct.

Treat the prediction as correct if it expresses the same meaning as the reference answer, even if it is shorter, uses aliases, omits redundant explanation, or formats numbers differently.
Treat the prediction as incorrect if it contradicts the reference answer, gives the wrong entity/value, or cannot be verified by the reference answer and supporting evidence.

# Question
{question}

# Supporting Evidence
{evidence}

# Reference Answer
{gold_answer}

# Model Prediction
{answer}

Output only "Correct" or "Incorrect".
"""




PROMPTS["simple_question_decomposition"] = """
---Role---
You are the natural-language Logical Execution Graph (LEG) planner for CoRProRAG.

---Goal---
Decompose the complex question into a complete logical chain of independent, single-hop natural-language sub-questions. This is the same logical decomposition used by text CoRProRAG; the executor, not the planner, decides whether text, table, knowledge-graph, or image evidence is needed.

---Output schema---
Return exactly one valid JSON object:
{{
  "reasoning_plan": "short plan",
  "sub_queries": [
    {{
      "query": "one single-hop natural-language sub-question",
      "has_dependency": false,
      "depends_on": []
    }}
  ]
}}

---Rules---
1. Use at most 6 sub-questions and make every sub-question single-hop: one directly retrievable fact or one directly observable visual fact.
2. If the original question is already single-hop, return one sub-question equal to it.
3. Use placeholders like {{answer_1}} only when a later sub-question needs an earlier answer.
4. Set has_dependency consistently with depends_on. Include every referenced {{answer_i}} in depends_on, including entity names used in a comparison.
5. Preserve every logical constraint from the original question, including visual predicates, names, dates, membership conditions, and comparison conditions. Keep poster/image questions explicitly visual; do not rewrite them as questions about the title text.
6. Do not output operation_type, modality, input_source, edge_type, expected_output_type, strategy, or tool names.
7. Do not explain how to search. Ask only the factual or visual sub-question that must be answered.

---Examples---
Question: What was the role in Nona Gaye films for the title that features a train on the poster?
{{
  "reasoning_plan": "Find Nona Gaye film candidates, identify which candidate title has a train on the poster, then retrieve the role.",
  "sub_queries": [
    {{"query": "What film titles are associated with Nona Gaye?", "has_dependency": false, "depends_on": []}},
    {{"query": "Which of {{answer_1}} has a poster that features a train?", "has_dependency": true, "depends_on": [1]}},
    {{"query": "What role did Nona Gaye play in {{answer_2}}?", "has_dependency": true, "depends_on": [2]}}
  ]
}}

Question: What is the person in the S. Cole TV title cover where she played Muriel doing?
{{
  "reasoning_plan": "Resolve the TV title from text first, then inspect the title cover.",
  "sub_queries": [
    {{"query": "What TV title did S. Cole play Muriel in?", "has_dependency": false, "depends_on": []}},
    {{"query": "In the cover image of {{answer_1}}, what is the person doing?", "has_dependency": true, "depends_on": [1]}}
  ]
}}

Question: What role did Vandit Bhatt play in the television show that has a poster featuring a white building with a domed roof?
{{
  "reasoning_plan": "Use the named person to get a small textual candidate set first, then use the poster clue to select the title, then retrieve the role.",
  "sub_queries": [
    {{"query": "What television shows did Vandit Bhatt appear in?", "has_dependency": false, "depends_on": []}},
    {{"query": "Which of {{answer_1}} has a poster featuring a white building with a domed roof?", "has_dependency": true, "depends_on": [1]}},
    {{"query": "What role did Vandit Bhatt play in {{answer_2}}?", "has_dependency": true, "depends_on": [2]}}
  ]
}}

---Real question---
Question: {query}

Return exactly one valid JSON object and nothing else.
"""

PROMPTS["simple_adaptive_corrector"] = """
---Role---
You are the Adaptive Corrector for natural-language CoRProRAG.

---Original question---
{original_query}

---Failed sub-question---
{sub_query}

---Previous answers---
{previous_answers}

---Failure information---
{failure_information}

---Task---
Decide whether the failed sub-question is already a single-hop factual or visual question.
- If it is single-hop, rewrite it to be specific and retrieval-friendly without changing the intended answer.
- If it is not single-hop, decompose it into independent, single-hop natural-language sub-questions.
- Preserve visual clues as visual facts, but never output modality labels, operations, strategies, or tool names.

Return exactly one JSON object:
{{
  "rationale": "short diagnosis",
  "is_single_hop": true,
  "reconstructed_query": "rewritten query when single-hop, otherwise empty",
  "sub_queries": [
    {{"query": "single-hop sub-question", "has_dependency": false, "depends_on": []}}
  ]
}}
"""

PROMPTS["simple_cor_final_answer_prompt"] = """
---Role---
You produce the final answer for the natural-language CoRProRAG pipeline.

---Original question---
{original_query}

---Logical plan---
{simple_plan}

---Sub-question answers---
{subquestion_results}

---Final multimodal SGN retrieval---
{final_retrieval}

---Instructions---
1. Answer the original question directly and concisely.
2. Connect the dependent sub-question answers according to the logical plan.
3. Use the final SGN evidence to verify or complete the chain.
4. Use only the supplied text, table, KG, image, and visual evidence.
5. If evidence is insufficient, answer "I don't know."
6. Return JSON only, without markdown.

{{
  "answer": "concise final answer",
  "rationale": "brief evidence-based reasoning"
}}
"""

PROMPTS["simple_retrieval_decision_prompt"] = """
---Role---
You choose a retrieval strategy for one simple CoR sub-question.

---Original question---
{original_query}

---Previous sub-question answers---
{previous_answers}

---Current sub-question---
{sub_question}

---Allowed strategies---
- text_only: use text/table/KG evidence only.
- image_only: retrieve KB/corpus images and inspect images only.
- text_then_image: first use text/table/KG evidence to narrow candidates, then retrieve/inspect images.
- image_then_text: first retrieve/inspect images to identify an entity, then use text/table/KG evidence.
- both_parallel: retrieve text/table/KG evidence and image evidence independently, then answer from both.
- direct_answer_if_enough: answer from previous sub-question answers only.

---Decision rules---
1. Choose text_then_image when a textual candidate set should narrow a visual check.
2. Choose image_then_text when a visual clue identifies an entity and the sub-question then asks for a textual attribute.
3. Choose image_only only when the answer is directly visual.
4. Choose text_only only when no visual evidence is needed.
5. Choose both_parallel if uncertain.
6. Keep text_query and image_query short. Do not force table/list membership into image_query if text_then_image is better.
7. Preserve the current sub-question's meaning. What a person holds or wears,
   colors, positions, and poster contents require image inspection. Movie titles
   and entity names do not establish those visual facts.
8. Use the current sub-question to choose tools; do not repeat unrelated later
   visual steps while retrieving a textual candidate list.
9. Keep rationale under 12 words.

Return exactly one JSON object:
{{
  "strategy": "text_only|image_only|text_then_image|image_then_text|both_parallel|direct_answer_if_enough",
  "text_query": "query for text/table/KG retrieval",
  "image_query": "query for KB/corpus image retrieval",
  "rationale": "under 12 words"
}}
"""


PROMPTS["simple_visual_observation_prompt"] = """
You are inspecting one candidate image for a multimodal QA sub-question.

Report the visible attribute or relation asked about. Text/table conditions
(credits, nominations, years, membership) are checked elsewhere: do not require
the image to prove them. Do not infer visible objects from the title/name.

Sub-question:
{sub_question}

Return exactly one JSON object:
{{
  "visual_observation": "concise observation relevant to the sub-question, or I don't know.",
  "rationale": "short reason"
}}
"""

PROMPTS["simple_visual_verification_prompt"] = """
You are a strict visual verifier for one candidate image.

Sub-question:
{sub_question}

Candidate entity/title:
{candidate}

Visual predicate to verify:
{visual_predicate}

Instructions:
1. Inspect the image only for the visual predicate.
2. Do not use world knowledge or text/table facts to answer yes.
3. Treat the candidate entity/title as an opaque label, not visual evidence.
4. Do not infer a feature from the candidate name itself.
5. Return "yes" only if the image clearly satisfies the predicate.
6. Return "no" if the predicate is clearly absent.
7. Return "uncertain" if the image is ambiguous or the predicate cannot be checked.
8. Keep observation under 20 words and rationale under 10 words.
9. Verify only visible properties. Do not require an image to prove nomination,
   film credits, dates, or table membership; those are separate textual facts.

Return exactly one JSON object:
{{
  "result": "yes|no|uncertain",
  "confidence": 0.0,
  "observation": "under 20 words",
  "rationale": "under 10 words"
}}
"""

PROMPTS["simple_visual_selector_disambiguation_prompt"] = """
---Role---
You resolve an ambiguous visual selector using existing visual checks.

---Sub-question---
{sub_question}

---Candidate visual checks---
{candidate_checks}

---Allowed candidates---
{allowed_candidates}

---Instructions---
1. Select exactly one allowed candidate if one best matches the visual predicate.
2. Use only the visual observations listed above.
3. Treat candidate names as opaque labels; do not prefer semantic name matches.
4. Prefer the observation that most explicitly shows all required visual parts.
5. If no candidate is clearly better, return an empty string.
6. Keep rationale under 12 words.

Return exactly one JSON object:
{{
  "selected_candidate": "one allowed candidate or empty string",
  "rationale": "under 12 words"
}}
"""

PROMPTS["simple_subquestion_answer_prompt"] = """
---Role---
You answer one simple CoR sub-question using only the provided evidence.

---Original question---
{original_query}

---Previous sub-question answers---
{previous_answers}

---Current sub-question---
{sub_question}

---Chosen retrieval strategy---
{chosen_strategy}

---Evidence---
{evidence}

---Instructions---
1. Answer the current sub-question, not the full original question unless they are the same.
2. Use only the provided evidence and previous answers.
3. Prefer evidence-backed entities or values.
4. Keep answer concise.
5. If the evidence is insufficient, answer "I don't know."
6. For a question selecting from prior candidates, return only members of that
   candidate set. Preserve all supported matches when multiple answers are asked.
   Return multiple entities as a JSON array, not a comma-separated string.
7. Use text/table evidence for membership and factual conditions, and visual
   observations for visible properties. A title is not evidence of image contents.
9. An observation about another named entity cannot establish a visual property of the requested entity.
8. Do not discard a visual observation because the image cannot independently
   prove a text/table fact already established by the supplied evidence.

Return exactly one valid JSON object:
{{
  "answer": "concise answer",
  "rationale": "short evidence-based reason",
  "confidence": 0.0
}}
"""




# Prompt for answering sub-queries (used by both dependent and independent handlers)
PROMPTS["answer_sub_query"] = """
---Role---
You are an expert at answering questions by extracting facts strictly and solely from the provided text.
---Context---
Below is the knowledge you must use to answer the sub-question. You cannot use any other information.
{context}
---Instruction---
Your task is to answer the "Sub-question to Answer" based exclusively on the provided "Context".
1.  Read the "Context" carefully to find the information that directly answers the "Sub-question to Answer".
2. Write the answer first. Keep rationale under 40 words; cite the source briefly without copying long passages.
3.  The "answer" should be a concise and direct response to the sub-question, extracting only the necessary information.
4.  If the "Context" does not contain the information to answer the question, you must state 'I don't know.' in the "answer" field and explain why the information is missing in the "rationale". Do not make up any information.
---Target response format---
Your response should be precise and direct, and your output must strictly follow the format below. Make sure your output is parsable by json in Python, and do not include ``` in the reply.
{{
    "answer": "<Only the direct answer; use a JSON array for multiple entities.>",
    "rationale": "<Brief evidence reference, at most 40 words.>"
}}
---Sub-question to Answer---
{sub_query}
"""


PROMPTS["classify_subquery"] = """
---Role---
You are an expert analyst specializing in problem evaluation and reconstruction. Your skill lies in determining whether a **sub-problem** is indivisible (single-hop) or requires further decomposition, and then taking appropriate action to optimize information retrieval. The original query is provided only as context to keep the sub-problem on track, not as the basis for your divisibility judgment.

---Instruction---
Your task is to examine a given sub-problem in relation to its original query and make a critical decision based solely on the sub-problem’s complexity:

1. If the sub-problem is indivisible (a single-hop question that can be directly answered through a single retrieval):
   - Carefully reconstruct the sub-problem to make it more specific and precise
   - Ensure the reconstructed version maintains the same intended answer
   - Focus on enhancing clarity and specificity to improve information retrieval

2. If the sub-problem is NOT indivisible (requires multiple steps or contains multiple questions):
   - Decompose it further into independent, single-hop sub-questions, following the format in "question_decomposition"
   - Apply the same principles from the original problem decomposition
   - Use appropriate dependency structure with placeholders when necessary

When evaluating, consider these criteria for an indivisible sub-problem:
- It seeks a single, specific fact or knowledge point
- It can be answered through direct retrieval without requiring inference chains
- It represents a fundamental question that cannot be meaningfully divided further
- It focuses on one distinct aspect of the original query

Return your analysis in the specified JSON format and do not include ``` in the reply.

---Example 1 (Indivisible Sub-Problem)---
Input data：
original query：What year did the region where Leuna was located come to an end?
the subquery you need to determine：Where is Leuna located?

Output data：
{{
  "rationale": "This sub-problem is indivisible because it seeks a single factual piece of information (Leuna's location) that cannot be meaningfully broken down further. It represents a fundamental first step in answering the original query about when a region ended. While already clear, I can reconstruct it to be more specific to improve retrieval of the most relevant historical information.",
  "is_single_hop": true,
  "reconstructed query": "In which historical region or country was the city of Leuna historically located?",
  "sub_queries": []
}}

---Example 2 (Divisible Sub-Problem)---
Input data：
original query：Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country where Sumardi was born?
the subquery you need to determine：Which country has a Commission of Truth and Friendship with the country where Sumardi was born?

Output data：
{{
  "rationale": "This sub-problem is not indivisible as it contains multiple logical steps. First, we need to determine where Sumardi was born, and then identify which country has a Commission of Truth and Friendship with that country. Breaking it down ensures each step represents a single retrievable fact.",
  "is_single_hop": false,
  "reconstructed query": "",
  "sub_queries": [
    {{
      "query": "Where was Sumardi born?",
      "has_dependency": false
    }},
    {{
      "query": "Which country has established a Commission of Truth and Friendship with {{answer_1}}?",
      "has_dependency": true,
      "depends_on": [1]
    }}
  ]
}}

---Example 3 (Divisible Sub-Problem)---
Input data：
original query：What is the capital of the country where Alfred Kurella hails from?
the subquery you need to determine：What is the capital of Alfred Kurella's home country?

Output data：
{{
  "rationale": "This sub-problem is not indivisible as it contains multiple logical steps. First, we need to determine where Alfred Kurella is from, and then find out what the capital of that country is. Breaking it down ensures each step represents a single retrievable fact.",
  "is_single_hop": false,
  "reconstructed query": "",
  "sub_queries": [
    {{
      "query": "Which country is Alfred Kurella from?",
      "has_dependency": false
    }},
    {{
      "query": "What is the capital of {{answer_1}}?",
      "has_dependency": true,
      "depends_on": [1]
    }}
  ]
}}

---real input data---
original query:{original_query}
the subquery you need to determine:{sub_query}
"""

PROMPTS["graphnavrag_prompt"] = """
---Role---
You are a helpful assistant.

---Context---
Use the following information to answer the question:
{context}

---Instruction---
1.  **Prioritize Context**: Base your answer primarily on the provided "Context". Your "rationale" must explain your reasoning step-by-step, showing how the context supports your conclusion.
2.  **Handle Insufficient Context**: If the context is not enough to answer fully, use your general knowledge to supplement it. In the "rationale", you must clearly distinguish between information from the context and information from your general knowledge.
3.  **Be Direct in Your Answer**: The "answer" field must be a concise and direct response to the question. You must always provide an answer.

---Target Format---
Your output must be a JSON object that can be parsed in Python. Do not include ``` in your reply.
{{
    "rationale": "<Detailed, step-by-step reasoning based on the provided context and sub-questions, explaining how the answer specifically addresses the Original Question. Show the logical links between the information found.>",
    "answer": "<Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>"
}}

---Original Question---
{query}
"""

PROMPTS["zero_shot_better"] = """
# Role
You are a helpful AI assistant on question answering.

# Task
Your task is to give your answer to the given question.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python. Do not include ``` in your reply.
{{
    "answer": <Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>
}}

# Question
{question}
"""

# ──────────────────────────────────────────────
# Image Modality Prompts (Image-Native CoRProRAG)
# ──────────────────────────────────────────────

PROMPTS["image_global_description"] = """
---Role---
You are a vision-language model capable of understanding and describing images in rich detail.

---Goal---
Given an image, provide a comprehensive global description of the entire scene.

---Steps---
1. Describe the overall setting, context, and atmosphere of the image.
2. Identify the key subjects (people, objects, text, charts, etc.) and their general positions.
3. Note any visible text, labels, captions, or data that appear in the image.
4. If the image contains a chart, table, or figure, describe its structure and key data points.
5. Provide a natural-language summary suitable for later use in a knowledge graph.

---Output Format---
Return ONLY the description text. No JSON, no markdown, no commentary.
"""

PROMPTS["query_image_understanding"] = """
---Role---
You are the visual grounding module of a multimodal retrieval system.

---Goal---
Given the original question and the question image, derive the visual sub-question
that must be answered from the image, then identify only the visual entities needed
for downstream KG retrieval.

---Question---
{query}

---Rules---
1. First infer a short visual sub-question, such as "Which food dishes are shown?" or "Which dog breeds are shown?".
2. Answer only that visual sub-question. Do NOT answer the original knowledge question.
3. Resolve phrases like "this food", "the food in the image", "this dog", "this bird", "this plant", "the shown object", or "people in the image" to the smallest necessary set of visible entities.
4. Prefer canonical entity names when visually identifiable, such as a person name, species, breed, food/dish, product, logo, place, chart/table title, or visible text.
5. If a visible entity is not identifiable, use one concise question-relevant category, e.g. "unknown dog breed" or "unknown food dish"; do not invent a name.
6. Exclude background, scene materials, colors, lighting, poses, clothing, floor, grass, sky, shadows, and other attributes from the entity list unless the visual sub-question explicitly asks about them.
7. Do not include text entities already named in the question unless they are also visible in the image and needed to resolve the visual referent.
8. Return at most 3 visual entities, ordered by answer relevance.

---Output Format---
Return exactly this plain text format, no JSON, no markdown, no commentary:
Visual sub-question: the image-only question you answered
Visual entities in the question image: entity1, entity2, entity3
Visual evidence: one compact sentence explaining how the image supports those entities.
"""

PROMPTS["visual_grounding_subquery"] = """
---Role---
You are the visual grounding executor in a multimodal multi-hop RAG system.

---Original Question---
{original_query}

---Visual Sub-question---
{visual_sub_query}

---Goal---
Look at the question image and answer ONLY the visual sub-question. This result will
be used as the answer to a dependency-graph node before KG retrieval.

---Rules---
1. Identify only the object(s), entity names, visible text, logo, species, breed, dish, person, place, chart element, or visual attribute required by the visual sub-question.
2. Do not list all visible objects. Ignore background and irrelevant objects.
3. Prefer canonical names when visually identifiable. If uncertain, return a concise question-relevant category instead of inventing a name.
4. Do not answer the downstream knowledge question; only resolve what the image contributes.
5. Return at most 3 visual entities, ordered by relevance.

---Output Format---
Return valid JSON only, without markdown:
{{
    "rationale": "A compact image-grounding explanation.",
    "visual_entities": ["entity1", "entity2"],
    "answer": "entity1 | entity2"
}}
"""

PROMPTS["image_entity_relation_extraction"] = """
---Goal---
Extract a compact, high-signal visual graph for question answering. Do not describe everything in the image.

---Context---
Image ID: {image_id}
Nearby Text: {text_context}

---Rules---
1. Create exactly one IMAGE entity named "IMAGE:{image_id}" with a one-sentence caption.
2. Extract at most 5 important visual entities. Prefer people, objects, logos, text blocks, charts, products, signs, and answer-relevant items.
3. Use IDs like "VISUAL_ENTITY:{image_id}:person_01" or "VISUAL_ENTITY:{image_id}:object_01". Never use descriptive names as IDs.
4. Keep visual entity descriptions under 18 words.
5. Add CONTAINS relationships from IMAGE to each visual entity.
6. Add at most 4 VISUAL_RELATION relationships between visual entities when clearly grounded.
7. Ignore background clutter, decorative objects, and uncertain details.

---Output Format---
Entity:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<short_description>)

Relationship:
("relationship"{tuple_delimiter}<src>{tuple_delimiter}<tgt>{tuple_delimiter}<short_description>{tuple_delimiter}<CONTAINS or VISUAL_RELATION>{tuple_delimiter}<strength>)

Return only records separated by {record_delimiter}. End with {completion_delimiter}.
"""

PROMPTS["image_continue_extraction"] = """MANY visual entities were missed in the last extraction. Add them below using the same format:
"""

PROMPTS["image_if_loop_extraction"] = """It appears some visual entities may have still been missed. Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["cross_modal_candidate_alignment"] = """
---Role---
You are an expert at cross-modal entity alignment. Your task is to determine whether a visual entity (from an image) matches a text entity (from a document).

---Goal---
For the given visual entity and its candidate text entities, decide whether there is an identity match.

---Rules---
1. Analyze the visual entity's description and the nearby text context.
2. For each candidate text entity, assess whether it refers to the SAME real-world person/object/location as the visual entity.
3. Return your decision with a confidence score.
4. If you are NOT confident (confidence < 0.6), choose "NO_MATCH".
5. Only return "IDENTITY" if you are highly confident (>= 0.85). For moderate confidence (0.6-0.85), use "DEPICTS".

---Input---
Visual Entity: {visual_entity_name}
Visual Description: {visual_description}
Nearby Text: {nearby_text}
Image Caption: {image_caption}

Candidates:
{candidates_json}

---Output Format---
Return exactly one valid JSON object and nothing else. Do not include ``` in the response.
The JSON object must use exactly these keys:
- visual_entity: string
- decision: string, either one of the candidate names or "NO_MATCH"
- edge_type: string, one of "IDENTITY", "DEPICTS", or ""
- confidence: number between 0.0 and 1.0
- evidence: string, at most 25 words
Do not explain your reasoning outside the JSON. Keep evidence brief and do not repeat yourself.
{{
    "visual_entity": "{visual_entity_name}",
    "decision": "<matched text entity name, or 'NO_MATCH'>",
    "edge_type": "IDENTITY | DEPICTS | \"\"",
    "confidence": <0.0 to 1.0>,
    "evidence": "<brief explanation, at most 25 words>"
}}
"""





PROMPTS["visual_answer_verification"] = """
---Role---
You are a multimodal model capable of verifying answers against image evidence.

---Goal---
Given a question, a proposed answer, and one or more relevant images, verify whether the answer is supported by visual evidence in the images.

---Steps---
1. Examine each image carefully.
2. Look for visual evidence that supports or contradicts the proposed answer.
3. Note any details in the images that are relevant to the question but not captured in the answer.
4. Return your verification with a confidence score.

---Input---
Question: {question}
Proposed Answer: {proposed_answer}
Reasoning Path: {reasoning_path}

---Output Format---
Return valid JSON. Do not include ``` in the response.
{{
    "verified": true | false,
    "confidence": <0.0 to 1.0>,
    "supporting_evidence": "<what in the images supports the answer>",
    "contradicting_evidence": "<what in the images contradicts the answer, if any>",
    "refined_answer": "<if the answer needs refinement based on visual evidence, provide the corrected answer; otherwise leave as is>"
}}
"""
