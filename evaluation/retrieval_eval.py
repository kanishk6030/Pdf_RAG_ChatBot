import json
import pandas as pd
def evaluate_retriever(retriever):

    with open(
        "evaluation/test_cases.json",
        "r"
    ) as f:
        test_cases = json.load(f)

    total_questions = len(test_cases)

    hit_count = 0

    total_recall = 0
    total_precision = 0
    total_f1 = 0

    results = []

    for test in test_cases:

        docs = retriever.invoke(
            test["question"]
        )

        retrieved_pages = {
            doc.metadata["page"] + 1
            for doc in docs
        }

        expected_pages = set(
            test["expected_pages"]
        )

        relevant_pages = (
            retrieved_pages
            &
            expected_pages
        )

        # Hit Rate

        if len(relevant_pages) > 0:
            hit_count += 1

        # Recall

        recall = (
            len(relevant_pages)
            /
            len(expected_pages)
        )

        total_recall += recall

        # Precision

        precision = (
            len(relevant_pages)
            /
            len(retrieved_pages)
        )

        total_precision += precision

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
            ) / (
            precision + recall
            )
        else:
            f1 = 0

        total_f1 += f1

        print("="*50)

        print(
            f"Question: {test['question']}"
        )

        print(
            f"Expected: {expected_pages}"
        )

        print(
            f"Retrieved: {retrieved_pages}"
        )

        print(
            f"Recall: {recall:.2f}"
        )

        print(
            f"Precision: {precision:.2f}"
        )  

        print(
            f"F1 Score: {f1:.2f}"
        )
        results.append({
            "Question": test["question"],
            "Recall": round(recall, 2),
            "Precision": round(precision, 2),
            "F1": round(f1, 2)
        })

    hit_rate = (
        hit_count
        /
        total_questions
    )

    avg_recall = (
        total_recall
        /
        total_questions
    )

    avg_precision = (
        total_precision
        /
        total_questions
    )

    avg_f1 = (
        total_f1    
        /
        total_questions
    )
    print("\nFINAL RESULTS")

    print(
        f"Hit Rate: {hit_rate:.2f}"
    )

    print(
        f"Average Recall: {avg_recall:.2f}"
    )

    print(
        f"Average Precision: {avg_precision:.2f}"
    )

    print(
        f"Average F1 Score: {avg_f1:.2f}"
    )

    df = pd.DataFrame(results)

    print("\nDetailed Results")
    print(df)
