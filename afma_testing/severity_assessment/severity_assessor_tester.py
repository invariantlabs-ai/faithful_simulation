import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import asyncio
import json
from scipy.stats import spearmanr
from src.afma.evaluation.severety_assessor import SideEffectSeverityAssessor
import litellm
from litellm.caching.caching import Cache

litellm.cache = Cache(type="disk")

async def main():
    # Load manual labels
    with open('afma_testing/severity_assessment/manual_labels.json') as f:
        manual_labels = json.load(f)

    # Initialize the severity assessor
    embedding_config = {
        "model": "text-embedding-3-small",
        "caching": True,
    }
    assessor = SideEffectSeverityAssessor(embedding_config)

    # Define the order of severity
    severity_order = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}

    all_tools = json.load(open('results/tools/all_tools.json'))

    tasks = []
    tool_contexts = []
    for toolset_name, toolset in all_tools.items():
        for tool in toolset['tools']:
            tool_name = tool['name']
            tool_info = f"{tool['name']}: {tool['description']}"
            
            manual_label = manual_labels.get(toolset_name, {}).get(tool_name)
            if not manual_label:
                continue

            tasks.append(assessor.assess_severity(tool_info))
            tool_contexts.append({'tool_name': tool_name, 'manual_label': manual_label, 'tool_info': tool_info})
    
    results = await asyncio.gather(*tasks)

    manual_scores = []
    assessed_scores = []
    output_results = []

    for i, (score, details) in enumerate(results):
        context = tool_contexts[i]
        tool_name = context['tool_name']
        manual_label = context['manual_label']
        assessed_label = details['best_match']

        manual_scores.append(severity_order[manual_label])
        assessed_scores.append(severity_order[assessed_label])

        result_item = {
            'tool_name': tool_name,
            'manual_label': manual_label,
            'assessed_label': assessed_label,
            'assessed_score': score,
            'details': details
        }
        output_results.append(result_item)

        print(f"Tool: {tool_name}")
        print(f"  Manual: {manual_label}")
        print(f"  Assessed: {assessed_label} (Score: {score:.2f})")
        print("-" * 20)


    # Calculate Spearman's rank correlation
    correlation, p_value = spearmanr(manual_scores, assessed_scores)
    print(f"\nSpearman's Rank Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Save results to a file
    with open('results/severity_assessor.json', 'w') as f:
        json.dump(output_results, f, indent=2)
    print("\nResults saved to results/severity_assessor.json")

if __name__ == "__main__":
    asyncio.run(main()) 