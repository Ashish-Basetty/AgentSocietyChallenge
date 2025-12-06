# Complete Workflow Documentation

## Overview

This project simulates Yelp user behavior using LLM-based agents. The goal is to predict how a real Yelp user would rate and review a specific business, given their historical profile and the business details.

---

## Inputs and Outputs

### Input (Task)
A JSON file containing:
```json
{
  "type": "user_behavior_simulation",
  "user_id": "wAo7casDFsbUR4O8Vb3u8A",
  "item_id": "cVoyA9wrdF5E8OroBahxCg"
}
```

- **user_id**: Unique identifier for the Yelp user to simulate
- **item_id**: Unique identifier for the business/restaurant to review

### Output (Prediction)
A dictionary containing:
```json
{
  "stars": 2.0,
  "review": "I had mixed feelings about my experience..."
}
```

- **stars**: Float (1.0, 2.0, 3.0, 4.0, or 5.0) - Predicted rating
- **review**: String - Predicted review text (max 512 characters)

### Ground Truth
Used for evaluation:
```json
{
  "stars": 2.0,
  "review": "I had mixed feelings about my experience with Mad Batter Bakery..."
}
```

---

## Complete Workflow

### Step 1: Initialization

```
Simulator Setup
├── Load dataset (dataset/)
│   ├── user.json (all user profiles)
│   ├── item.json (all business profiles)
│   └── review.json (all reviews)
├── Load tasks (example/track1/yelp/tasks/*.json)
├── Load ground truth (example/track1/yelp/groundtruth/*.json)
├── Set Agent Class (e.g., TOTSimulationAgent)
└── Set LLM (e.g., GeminiLLM)
```

### Step 2: Task Execution Flow

For each task:

```
1. Create Agent Instance
   └── Agent initialized with LLM

2. Set Task
   └── agent.insert_task(task) 
   └── task = {"user_id": "...", "item_id": "..."}

3. Execute Agent Workflow
   └── output = agent.workflow()
```

### Step 3: Agent Workflow (TOTSimulationAgent)

The agent workflow consists of multiple phases:

#### Phase 1: Planning
```
Planning Module (e.g., PlanningBaseline)
├── Input: task_description = {"user_id": "...", "item_id": "..."}
└── Output: plan = [
    {
        'description': 'First I need to find user information',
        'reasoning instruction': 'None',
        'tool use instruction': {user_id}
    },
    {
        'description': 'Next, I need to find business information',
        'reasoning instruction': 'None',
        'tool use instruction': {item_id}
    }
]
```

The planning module breaks down the task into subtasks that determine what information needs to be gathered.

#### Phase 2: Data Retrieval
```
For each subtask in plan:
├── If subtask contains 'user':
│   └── user_data = interaction_tool.get_user(user_id)
│       └── Returns: Complete user profile with review history
│
├── If subtask contains 'business':
│   └── business_data = interaction_tool.get_item(item_id)
│       └── Returns: Complete business profile with details
│
└── Get additional context:
    ├── reviews_item = interaction_tool.get_reviews(item_id)
    │   └── All reviews for this business
    │
    └── reviews_user = interaction_tool.get_reviews(user_id)
        └── User's past reviews (for style consistency)
```

The following data is retrieved:
- User Profile: Name, location, review history, preferences, rating patterns
- Business Profile: Name, category, location, attributes, price range
- Business Reviews: What other users said about this business
- User Reviews: The user's past review style and preferences

#### Phase 3: Memory Storage
```
Memory Module (e.g., MemoryDILU)
├── Store business reviews in memory
│   └── For each review: memory(f'review: {review_text}')
│
└── Retrieve similar user review
    └── review_similar = memory(f'{reviews_user[0]["text"]}')
    └── Uses similarity search to find relevant context
```

The memory module stores and retrieves relevant context to improve prediction accuracy.

#### Phase 4: Reasoning
```
Reasoning Module (e.g., ReasoningTOT)
├── Input: task_description (comprehensive prompt)
└── Output: LLM-generated rating and review
```

This phase generates the final rating and review prediction.

---

## The Complete Prompt

The prompt sent to the LLM includes:

### Context Provided

1. User Identity
   ```
   "You are a real human user on Yelp, a platform for crowd-sourced business reviews. 
   Here is your Yelp profile and review history: {user_data}"
   ```

2. Business Information
   ```
   "You need to write a review for this business: {business_data}"
   ```

3. Other Users' Reviews
   ```
   "Others have reviewed this business before: {review_similar}"
   ```
   (Retrieved from memory using similarity search)

### Instructions

```
Please analyze the following aspects carefully:
1. Based on your user profile and review style, what rating would you give this business? 
   Remember that many users give 5-star ratings for excellent experiences that exceed 
   expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.

2. Given the business details and your past experiences, what specific aspects would you 
   comment on? Focus on the positive aspects that make this business stand out or negative 
   aspects that severely impact the experience.

3. Consider how other users might engage with your review in terms of:
   - Useful: How informative and helpful is your review?
   - Funny: Does your review have any humorous or entertaining elements?
   - Cool: Is your review particularly insightful or praiseworthy?
```

### Requirements

```
- Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
- If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
- If the business fails significantly in key areas, consider giving a 1-star rating
- Review text should be 2-4 sentences, focusing on your personal experience and emotional response
- Maintain consistency with your historical review style and rating patterns
- Focus on specific details about the business rather than generic comments
- Be generous with ratings when businesses deliver quality service and products
- Be critical when businesses fail to meet basic standards
```

### Output Format

```
Format your response exactly as follows:
stars: [your rating]
review: [your review]
```

### Few-Shot Examples (for Reasoning Module)

The reasoning module also includes examples:
```
Example 1:
stars: 1.0
review: I had high hopes for the Masters Inn Fairgrounds, but my experience was a major 
letdown. The room was infested with roaches, and the furniture was old and falling apart...

Example 2:
stars: 3.0
review: I visited Arizona Bug Doctor for pest control services, and my experience was mixed. 
On the positive side, the technicians were knowledgeable and thorough...
```

---

## Reasoning Module Details

Different reasoning modules process this prompt differently:

### ReasoningIO
Uses direct input-output mapping with a simple prompt that includes examples.

### ReasoningCOT (Chain of Thought)
Adds step-by-step reasoning instructions to encourage explicit problem-solving.

### ReasoningTOT (Tree of Thoughts)
Generates three candidate responses and uses a voting mechanism to select the best one. This is the most robust approach.

### ReasoningSelfRefine
Generates an initial response, then refines it based on feedback.

### ReasoningStepBack
First extracts abstract principles, then applies them to the specific task.

---

## Evaluation

After predictions are generated:

### Preference Estimation (Rating Accuracy)
```
RMSE between predicted and actual stars
preference_estimation = 1 - (average absolute error / 5)
```

### Review Generation (Review Quality)
```
Combined metrics:
├── Sentiment Error (25%): VADER sentiment analysis comparison
├── Emotion Error (25%): RoBERTa emotion classification comparison
└── Topic Error (50%): SentenceTransformer embedding similarity

review_generation = 1 - (sentiment_error * 0.25 + emotion_error * 0.25 + topic_error * 0.5)
```

### Overall Quality
```
overall_quality = (preference_estimation + review_generation) / 2
```

---

## Key Components

### 1. Planning Module
Breaks the task into subtasks. Available options include PlanningBaseline, PlanningIO, PlanningTD, PlanningDEPS, and others. PlanningTD performs best with an overall quality score of 0.7453.

### 2. Reasoning Module
Generates the actual prediction. Available options include ReasoningTOT, ReasoningIO, ReasoningCOT, and others. ReasoningTOT is the fastest while maintaining the same quality.

### 3. Memory Module
Stores and retrieves relevant context. The primary option is MemoryDILU, which uses similarity search to find relevant past reviews.

### 4. Interaction Tool
Provides access to the dataset through the following methods:
- `get_user(user_id)` returns the user profile
- `get_item(item_id)` returns the business profile
- `get_reviews(item_id/user_id)` returns reviews

---

## Complete Data Flow Diagram

```
[Task Input]
    ↓
[Planning Module]
    ↓ (generates plan)
[Execute Plan: Retrieve Data]
    ├──→ get_user() → User Profile
    ├──→ get_item() → Business Profile
    ├──→ get_reviews(item_id) → Business Reviews
    └──→ get_reviews(user_id) → User's Past Reviews
    ↓
[Memory Module]
    ├──→ Store business reviews
    └──→ Retrieve similar user review (via similarity search)
    ↓
[Construct Prompt]
    ├── User profile & history
    ├── Business details
    ├── Other users' reviews (from memory)
    └── Instructions & examples
    ↓
[Reasoning Module]
    ├──→ Send prompt to LLM
    ├──→ Generate candidate(s)
    └──→ Parse output (stars + review)
    ↓
[Post-Processing]
    ├── Extract stars from "stars: X.X"
    ├── Extract review from "review: ..."
    └── Truncate review to 512 chars
    ↓
[Output]
    {
        "stars": 2.0,
        "review": "..."
    }
    ↓
[Evaluation]
    ├── Compare stars (RMSE)
    └── Compare reviews (sentiment + emotion + topic)
```

---

## Key Insights

### What Makes Predictions Accurate

Several factors contribute to prediction accuracy:

1. User Context: Understanding the user's historical preferences and review style
2. Business Context: Knowing the business details, category, and attributes
3. Social Context: Seeing what other users said about the business
4. Memory: Retrieving similar past experiences from memory
5. Reasoning: Multi-step reasoning to align user preferences with business attributes

### Why Planning Matters

The planning module determines what information to gather. Better planning leads to better context, which in turn leads to better predictions. PlanningTD performs best because it explicitly considers temporal dependencies between subtasks.

### Why Reasoning Matters Less

All reasoning modules achieve similar quality (0.5979), so execution speed becomes the main differentiator. ReasoningTOT is the fastest at 1.51 seconds while maintaining the same quality.

---

## Example Walkthrough

### Input Task
```json
{
  "user_id": "wAo7casDFsbUR4O8Vb3u8A",
  "item_id": "cVoyA9wrdF5E8OroBahxCg"
}
```

### What Happens

1. Planning: The system decides to fetch user information first, then business information.

2. Data Retrieval:
   - Retrieves the user profile including name, location, past 50+ reviews, and favorite categories
   - Retrieves the business profile for "Mad Batter Bakery" including category (Bakery), location, and attributes
   - Retrieves all reviews for "Mad Batter Bakery" (30+ reviews from other users)
   - Retrieves the user's past reviews to understand their writing style

3. Memory:
   - Stores all business reviews in memory
   - Searches for the user's most similar past review

4. Prompt Construction:
   ```
   You are [User Name], a Yelp user in [Location]. 
   Your past reviews show you value [preferences]...
   
   You need to review: Mad Batter Bakery
   [Business details: category, location, attributes...]
   
   Others have reviewed this: [Summary of other reviews]
   
   Based on your style and preferences, what would you rate and review?
   ```

5. LLM Reasoning:
   - Analyzes whether this user likes bakeries
   - Considers what other reviews say about the business
   - Matches how the business aligns with user preferences
   - Generates the rating and review

6. Output:
   ```json
   {
     "stars": 2.0,
     "review": "I had mixed feelings about my experience..."
   }
   ```

7. Evaluation:
   - Compares predicted stars (2.0) with actual stars (2.0)
   - Compares review sentiment, emotion, and topic similarity

---

## Key Takeaways

1. Input: Just `user_id` and `item_id`
2. Context: Retrieved from dataset via interaction tool
3. Planning: Determines what information to gather
4. Memory: Stores and retrieves relevant context
5. Reasoning: LLM generates prediction using all context
6. Output: Rating (1-5 stars) + Review text
7. Evaluation: Measures rating accuracy and review quality

The system answers the question: Given this user's history and this business, how would this user realistically rate and review it?

