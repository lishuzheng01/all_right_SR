# -*- coding: utf-8 -*-
"""
Command-line interface for sisso_py.
(Placeholder for future implementation using argparse or click)
"""
import argparse

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SISSO Regressor Command Line Interface")
    
    parser.add_argument("input_file", type=str, help="Path to the input data file (e.g., CSV).")
    parser.add_argument("target_column", type=str, help="Name of the target variable column.")
    parser.add_argument("-o", "--output", type=str, default="sisso_report.json", help="Path to save the output report JSON.")
    parser.add_argument("-k", type=int, default=2, help="Max complexity K for feature generation.")
    parser.add_argument("--max-terms", type=int, default=3, help="Max number of terms in the final model.")
    
    args = parser.parse_args()
    
    print("CLI is a placeholder. To be implemented.")
    print(f"Input file: {args.input_file}")
    print(f"Target column: {args.target_column}")
    print(f"Output file: {args.output}")
    
    # --- Example of future implementation ---
    # from sisso_py.model import SissoRegressor
    # from sisso_py.io import load_from_pandas
    # from sisso_py.io.export import export_to_json
    # import pandas as pd
    
    # df = pd.read_csv(args.input_file)
    # X, y = load_from_pandas(df, args.target_column)
    
    # model = SissoRegressor(K=args.k, so_max_terms=args.max_terms)
    # model.fit(X, y)
    
    # export_to_json(model, args.output)
    # print(f"Report saved to {args.output}")

if __name__ == '__main__':
    main()
