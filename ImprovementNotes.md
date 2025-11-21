# Baseline results
We used the provided structure of simulator and example baseline model to come up with our initial results, with no examples and a relatively simple agent structure. We got:
{
    "type": "simulation",
    "metrics": {
        "preference_estimation": 0.42666666666666664,
        "review_generation": 0.60427244149902,
        "overall_quality": 0.5154695540828433
    },
    "data_info": {
        "evaluated_count": 30,
        "original_simulation_count": 30,
        "original_ground_truth_count": 400
    }
}

# Improving reasoning

We first tried implementing a  *Tree of Thought (TOT)* system to improve responses. One other possible way to improve reasoning was to provide concrete examples for the agent. Using the ground truth data, we decided to provide a few hardcoded examples based on past responses. Using the last two yelp ground truth examples, we got the following results:
{
    "type": "simulation",
    "metrics": {
        "preference_estimation": 0.6533333333333333,
        "review_generation": 0.5875791056496582,
        "overall_quality": 0.6204562194914958
    },
    "data_info": {
        "evaluated_count": 30,
        "original_simulation_count": 30,
        "original_ground_truth_count": 400
    }
}

# Memory module experiments

To understand how different long-term memory strategies affect the TOT agent, we ran five-task smoke tests for each memory module variant (all other components identical). Results are below:

| Memory type | Preference Est. | Review Gen. | Overall Quality | Tasks |
|-------------|-----------------|-------------|-----------------|-------|
| DILU        | 0.6400          | 0.6160      | 0.6280          | 5     |
| Generative  | 0.6800          | 0.6157      | 0.6478          | 5     |
| TP          | 0.6800          | 0.6097      | 0.6448          | 5     |
| Voyager     | 0.7200          | 0.6083      | **0.6642**      | 5     |

**Takeaways**
- Voyager-style summarized memory delivered the best overall quality despite slightly weaker review generation, likely because the higher preference score carried more weight.
- Pure DILU memory underperformed every other option, so we should avoid relying on it without additional retrieval filtering.
- Generative scoring and TP planning were competitive but still a few points behind Voyager on this small sample.

**Next steps**
- Re-run Voyager vs. Generative on ≥30 tasks to confirm the ordering (current tests are noisy at only five tasks).
- Investigate hybrid approach (Voyager summaries for retrieval + Generative importance scoring) to combine high preference accuracy with richer contextual cues.
