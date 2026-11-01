"""Show a campaign's resource report.

    python -m research.resource_accounting --campaign-id ID [--prices configs/prices.json] [--json]
"""

import argparse
import json

from agent.experiments import FEEDBACK_VERSIONS
from research.resource_accounting.accounting import (
    campaign_resource_report,
    format_report,
    load_prices,
)


def main():
    parser = argparse.ArgumentParser(description="GoTrain campaign resource report")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--prices", help="JSON file with explicitly configured USD prices.")
    parser.add_argument("--feedback", choices=FEEDBACK_VERSIONS,
                        help="E-profile; detected from experiment logs when omitted.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()
    report = campaign_resource_report(
        args.campaign_id, load_prices(args.prices), version=args.feedback, seed=args.seed
    )
    print(json.dumps(report, indent=2) if args.json else format_report(report))


if __name__ == "__main__":
    main()
