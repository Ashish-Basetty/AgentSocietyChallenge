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

We first tried implementing a  *Tree of Thought (TOT)* system to improve responses. Simply replacing the baseline reasoning model with the Tree of though model gave us:
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
This gave us no initial discernible improvement, but one possible way to improve reasoning was to provide concrete examples for the agent. Using the ground truth data, we decided to provide a few hardcoded examples based on past responses. Using the last two yelp ground truth examples, we got the following results:
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

One modification to the *Tree of Thought* system was to immediately return a result if 'think' was part of the text. This shortcut sped up responses, but **COMPLETE**