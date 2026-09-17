import os
from datetime import datetime
import json
from collections import Counter

from llm.llm_output import ExtendedEncoder


def save_results_to_json(res, args, problem, all_evaluated=None, output_dir="results"):
    """Save the optimization results to a JSON file."""
    
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "seed": args.seed,
        "algorithm": args.algorithm,
        "population_size": args.n,
        "generations": args.i,
        "sut": args.sut,
        "archive_threshold": args.archive_threshold if args.algorithm == "nsga2d" else "N/A",
        "weight_clarity": args.weight_clarity,
        "weight_request_orientedness": args.weight_request_orientedness,
        "execution_time_seconds": res.exec_time,
        "problem_name": problem.problem_name if hasattr(problem, 'problem_name') else "Unknown",
        "timestamp": datetime.now().isoformat(),
        "actual_generations_completed": len(res.history) if hasattr(res, 'history') else args.i,
        "max_time": args.max_time,
    }
    
    # extract individuals from final population
    individuals_data = []
    final_population = res.opt

    if final_population is not None and len(final_population) > 0:
        for i, individual in enumerate(final_population):
            conversation = individual.X[0]
            fitness_scores = individual.F.tolist() if hasattr(individual.F, 'tolist') else list(individual.F)
            
            is_critical = bool(individual.CB) if hasattr(individual, 'CB') and individual.CB is not None else None
            
            # Extract full judge output and other scores
            # judge_output = {}
            other_scores = {}
            if hasattr(individual, 'SO') and individual.SO:
                # if hasattr(individual.SO, 'judge_output'):
                #     judge_output = individual.SO.judge_output
                if hasattr(individual.SO, 'other'):
                    other_scores = individual.SO.other

            conversation_dict = conversation.to_dict()

            features_dict = {}
            if hasattr(problem, "feature_handler") and problem.feature_handler and getattr(conversation, "ordinal_vars", None):
                try:
                    features_dict = problem.feature_handler.map_numerical_scores_to_labels(conversation.ordinal_vars)
                except Exception:
                    features_dict = {}

            individual_data = {
                "individual_id": i + 1,
                "fitness_scores": {
                    "clarity_request_orientedness": fitness_scores[0] if len(fitness_scores) > 0 else None,
                    "efficiency": fitness_scores[1] if len(fitness_scores) > 1 else None,
                    "average_effectiveness": fitness_scores[2] if len(fitness_scores) > 2 else None,
                },
                "is_critical": is_critical,
                "num_turns": len(conversation),
                "features_dict": features_dict,
                "other_scores": other_scores,
                "conversation": conversation_dict
            }
            
            individuals_data.append(individual_data)

    # Collect all evaluated individuals from history
    all_evaluated_data = []
    critical_count = 0
    turn_stats = Counter()  # Use Counter for cleaner counting

    # Try to get population from history first, then fall back to res.opt
    populations_to_process = []
    
    if hasattr(res, 'history') and res.history:
        for gen_idx, history_entry in enumerate(res.history):
            pop = history_entry.pop if hasattr(history_entry, 'pop') else None
            if pop is not None:
                populations_to_process.append((gen_idx, pop))
    
    # Fallback: if no history, use the final optimized population (res.opt)
    if not populations_to_process and res.opt is not None and len(res.opt) > 0:
        populations_to_process.append((0, res.opt))
    
    individual_counter = 0
    for gen_idx, population in populations_to_process:
        for ind in population:
            conversation = ind.X[0]
            fitness_scores = ind.F.tolist() if hasattr(ind.F, 'tolist') else list(ind.F)
            
            is_critical = bool(ind.CB) if hasattr(ind, 'CB') and ind.CB is not None else None
            
            if is_critical:
                critical_count += 1
            
            n_turns = len(conversation)
            turn_stats[n_turns] += 1  # Count by actual turn number

            # judge_output = {}
            other_scores = {}
            if hasattr(ind, 'SO') and ind.SO:
                # if hasattr(ind.SO, 'judge_output'):
                #     judge_output = ind.SO.judge_output
                if hasattr(ind.SO, 'other'):
                    other_scores = ind.SO.other
            
            conversation_dict = conversation.to_dict()

            features_dict = {}
            if hasattr(problem, "feature_handler") and problem.feature_handler and getattr(conversation, "ordinal_vars", None):
                try:
                    features_dict = problem.feature_handler.map_numerical_scores_to_labels(conversation.ordinal_vars)
                except Exception:
                    features_dict = {}

            all_eval_data = {
                "individual_id": individual_counter + 1,
                "generation": gen_idx,
                "fitness_scores": {
                    "clarity_request_orientedness": fitness_scores[0] if len(fitness_scores) > 0 else None,
                    "efficiency": fitness_scores[1] if len(fitness_scores) > 1 else None,
                    "average_effectiveness": fitness_scores[2] if len(fitness_scores) > 2 else None,
                },
                "is_critical": is_critical,
                "num_turns": n_turns,
                "features_dict": features_dict,
                # "judge_output": {
                #     "justification_clarity": judge_output.get("justification_clarity", "N/A"),
                #     "justification_request_orientedness": judge_output.get("justification_request_orientedness", "N/A"),
                #     "detailed_scores": judge_output.get("detailed_scores", {}),
                # },
                "other_scores": other_scores,
                "conversation": conversation_dict
            }
            
            all_evaluated_data.append(all_eval_data)
            individual_counter += 1

    # Convert turn_stats Counter to a sorted dictionary with descriptive keys
    turn_distribution = {f"{k}_turns": v for k, v in sorted(turn_stats.items())}
    
    # Compute additional statistics
    turn_counts = list(turn_stats.keys())
    avg_turns = sum(k * v for k, v in turn_stats.items()) / individual_counter if individual_counter > 0 else 0
    min_turns = min(turn_counts) if turn_counts else 0
    max_turns = max(turn_counts) if turn_counts else 0

    results_data = {
        "metadata": metadata,
        "statistics": {
            "turn_distribution": turn_distribution,
            "total_evaluated": individual_counter,
            "critical_count": critical_count,
            "critical_rate": critical_count / individual_counter if individual_counter > 0 else 0,
            "avg_turns": round(avg_turns, 2),
            "min_turns": min_turns,
            "max_turns": max_turns,
        },
        "all_evaluated_conversations": all_evaluated_data,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mt_test_results_{args.algorithm}_{args.n}n_{args.i}i_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    print(f"Saving overview to: {os.path.abspath(filepath)}")
    
    # save to JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, cls=ExtendedEncoder)
        return filepath
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        return None