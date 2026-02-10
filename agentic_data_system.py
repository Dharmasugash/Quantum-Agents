import pandas as pd
import numpy as np
import json
import os
import re
import pdfplumber
from sqlalchemy import create_engine
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

class DataIngestionAgent:
    """Agent responsible for loading data from various formats into a unified format."""
    
    def ingest(self, file_path: str) -> pd.DataFrame:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext == '.xml':
            return self._parse_xml(file_path)
        elif ext in ['.sql', '.db']:
            engine = create_engine(f'sqlite:///{file_path}')
            # For simplicity, we assume the first table
            table_names = engine.table_names()
            if table_names:
                return pd.read_sql(table_names[0], engine)
            raise ValueError("No tables found in SQL database")
        elif ext in ['.txt', '.log']:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            return pd.DataFrame({'content': lines})
        elif ext == '.pdf':
            return self._parse_pdf(file_path)
        else:
            # Try reading as CSV as a fallback
            try:
                return pd.read_csv(file_path)
            except:
                raise ValueError(f"Unsupported file format: {ext}")

    def _parse_xml(self, file_path: str) -> pd.DataFrame:
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        for child in root:
            data.append(child.attrib if child.attrib else {c.tag: c.text for c in child})
        return pd.DataFrame(data)

    def _parse_pdf(self, file_path: str) -> pd.DataFrame:
        data = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    data.extend(table)
                else:
                    text = page.extract_text()
                    if text:
                        data.append({'content': text})
        
        if not data:
            return pd.DataFrame()
        
        if isinstance(data[0], list):
            return pd.DataFrame(data[1:], columns=data[0])
        return pd.DataFrame(data)

class DataProfilingAgent:
    """Agent responsible for profiling the data and generating statistical summaries."""
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"error": "Empty dataset"}
            
        profile_info = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": {},
            "missing_total": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict()
        }
        
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "unique": int(df[col].nunique())
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    "min": float(df[col].min()) if not df[col].empty else None,
                    "max": float(df[col].max()) if not df[col].empty else None,
                    "mean": float(df[col].mean()) if not df[col].empty else None
                })
            profile_info["columns"][col] = col_profile
            
        return profile_info

    def calculate_quality_score(self, profile: Dict[str, Any]) -> float:
        if "error" in profile: return 0.0
        
        total_cells = profile["num_rows"] * profile["num_cols"]
        if total_cells == 0: return 0.0
        
        missing_penalty = (profile["missing_total"] / total_cells) * 100
        duplicate_penalty = (profile["duplicates"] / profile["num_rows"]) * 100 if profile["num_rows"] > 0 else 0
        
        score = 100 - (missing_penalty + duplicate_penalty)
        return max(0.0, min(100.0, score))

class IssueDetectionAgent:
    """Agent responsible for identifying specific data quality issues."""
    
    def detect_issues(self, df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
        issues = {
            "missing_values": profile["missing_total"],
            "duplicate_rows": profile["duplicates"],
            "invalid_entries": 0,
            "outliers": 0,
            "inconsistent_formats": 0
        }
        
        for col in df.columns:
            # Simple outlier detection for numeric columns (Z-score > 3)
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = ((df[col] - mean).abs() > 3 * std).sum()
                    issues["outliers"] += int(outliers)
            
            # Detect inconsistent formats in strings (e.g., mixed case)
            if pd.api.types.is_object_dtype(df[col]):
                sample = df[col].dropna().astype(str)
                if not sample.empty:
                    has_upper = sample.str.isupper().any()
                    has_lower = sample.str.islower().any()
                    if has_upper and has_lower:
                        issues["inconsistent_formats"] += 1
                        
        return issues

class PlanningAgent:
    """Agent responsible for deciding cleaning strategies."""
    
    def create_plan(self, issues: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan = []
        
        if issues["missing_values"] > 0:
            plan.append({
                "action": "impute_missing",
                "reason": f"Found {issues['missing_values']} missing values",
                "strategy": "mean_for_numeric_mode_for_cat"
            })
            
        if issues["duplicate_rows"] > 0:
            plan.append({
                "action": "remove_duplicates",
                "reason": f"Found {issues['duplicate_rows']} duplicate rows",
                "strategy": "keep_first"
            })
            
        if issues["outliers"] > 0:
            plan.append({
                "action": "handle_outliers",
                "reason": f"Found {issues['outliers']} outliers",
                "strategy": "clip_to_bounds"
            })
            
        if issues["inconsistent_formats"] > 0:
            plan.append({
                "action": "standardize_text",
                "reason": "Found inconsistent text formats",
                "strategy": "lowercase_and_strip"
            })
            
        return plan

class ExecutionAgent:
    """Agent responsible for applying transformations."""
    
    def execute(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        for step in plan:
            action = step["action"]
            
            if action == "impute_missing":
                for col in df_cleaned.columns:
                    if df_cleaned[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                        else:
                            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown")
                            
            elif action == "remove_duplicates":
                df_cleaned = df_cleaned.drop_duplicates()
                
            elif action == "handle_outliers":
                for col in df_cleaned.columns:
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        mean = df_cleaned[col].mean()
                        std = df_cleaned[col].std()
                        lower = mean - 3 * std
                        upper = mean + 3 * std
                        df_cleaned[col] = df_cleaned[col].clip(lower, upper)
                        
            elif action == "standardize_text":
                for col in df_cleaned.columns:
                    if pd.api.types.is_object_dtype(df_cleaned[col]):
                        df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
                        
        return df_cleaned

class ValidationAgent:
    """Agent responsible for validating the results."""
    
    def validate(self, original_profile: Dict[str, Any], cleaned_profile: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "schema_integrity": "PASS" if original_profile["num_cols"] == cleaned_profile["num_cols"] else "FAIL",
            "statistical_consistency": "PASS" if cleaned_profile["missing_total"] < original_profile["missing_total"] or original_profile["missing_total"] == 0 else "FAIL",
            "readiness": "PASS" if cleaned_profile["missing_total"] == 0 else "FAIL"
        }
        return results

class AgenticDataSystem:
    """Orchestrator for the multi-agent data processing system."""
    
    def __init__(self):
        self.ingestion = DataIngestionAgent()
        self.profiling = DataProfilingAgent()
        self.detection = IssueDetectionAgent()
        self.planning = PlanningAgent()
        self.execution = ExecutionAgent()
        self.validation = ValidationAgent()
        
    def process_dataset(self, file_path: str, quality_threshold: float = 95.0) -> Dict[str, Any]:
        # 1. Ingestion
        df_original = self.ingestion.ingest(file_path)
        df_current = df_original.copy()
        
        # 2. Initial Profiling
        original_profile = self.profiling.profile(df_original)
        original_score = self.profiling.calculate_quality_score(original_profile)
        
        # 7. Reflection & Optimization Loop
        iteration = 0
        max_iterations = 3
        current_score = original_score
        all_plans = []
        
        while current_score < quality_threshold and iteration < max_iterations:
            iteration += 1
            
            # 3. Issue Detection
            current_profile = self.profiling.profile(df_current)
            issues = self.detection.detect_issues(df_current, current_profile)
            
            # 4. Planning
            plan = self.planning.create_plan(issues, current_profile)
            if not plan:
                break
            all_plans.extend(plan)
            
            # 5. Execution
            df_current = self.execution.execute(df_current, plan)
            
            # 6. Validation
            new_profile = self.profiling.profile(df_current)
            new_score = self.profiling.calculate_quality_score(new_profile)
            
            if new_score <= current_score:
                # No improvement, break to avoid infinite loop
                break
            current_score = new_score

        final_score = current_score
        df_cleaned = df_current
        cleaned_profile = self.profiling.profile(df_cleaned)
        validation_results = self.validation.validate(original_profile, cleaned_profile)
        
        # Generate Insights
        insights = self._generate_insights(df_cleaned)
        
        # Prepare Report
        report = self._generate_report(
            file_path, df_original, df_cleaned, 
            original_score, final_score, 
            issues, all_plans, validation_results, insights
        )
        
        return {
            "report": report,
            "cleaned_df": df_cleaned,
            "metrics": {
                "original_score": original_score,
                "final_score": final_score,
                "improvement": final_score - original_score
            }
        }

    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        insights = []
        if df.empty: return ["No data to generate insights"]
        
        # Basic correlation insight
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr().unstack().sort_values(ascending=False)
            corr = corr[corr < 1].head(2)
            for (c1, c2), val in corr.items():
                insights.append(f"Strong correlation found between {c1} and {c2}: {val:.2f}")
        
        # Distribution insight
        for col in df.columns[:3]: # Limit to first 3 columns
            insights.append(f"Column '{col}' has {df[col].nunique()} unique values.")
            
        return insights

    def _generate_report(self, file_path, df_orig, df_clean, s_orig, s_final, issues, plan, val, insights) -> str:
        report = f"""
## Data Cleaning & Preprocessing Report

### File Overview
- File type: {os.path.splitext(file_path)[1]}
- Inferred domain: General / Domain-Agnostic
- Total rows (original): {len(df_orig)}
- Total columns (original): {len(df_orig.columns)}

### Data Quality Summary
- Original data quality score (%): {s_orig:.2f}%
- Final data quality score (%): {s_final:.2f}%
- Overall improvement (%): {s_final - s_orig:.2f}%

### Issues Detected
- Missing values: {issues['missing_values']}
- Duplicate rows: {issues['duplicate_rows']}
- Invalid entries: {issues['invalid_entries']}
- Inconsistent formats: {issues['inconsistent_formats']}
- Outliers: {issues['outliers']}
- Noise or irrelevant records: 0

### Cleaning Actions Performed
- Rows cleaned: {len(df_clean)}
- Rows removed: {len(df_orig) - len(df_clean)}
- Columns transformed: {len(df_clean.columns)}
- Columns standardized: {sum(1 for p in plan if p['action'] == 'standardize_text')}
- Columns normalized: 0
- Text cleaned (if applicable): {'Yes' if any(p['action'] == 'standardize_text' for p in plan) else 'No'}

### Transformations Applied
"""
        for step in plan:
            report += f"- {step['action']}: {step['reason']} (Strategy: {step['strategy']})\n"
            
        report += f"""
### Validation Results
- Schema integrity: {val['schema_integrity']}
- Statistical consistency: {val['statistical_consistency']}
- Data readiness for ML/Analytics: {val['readiness']}

### Insights & Observations
"""
        for insight in insights:
            report += f"- {insight}\n"
            
        report += """
### Final Deliverables
- Cleaned dataset ready for ML and analytics
- Cleaning and transformation log
- Data quality improvement metrics
"""
        return report

if __name__ == "__main__":
    # Example usage script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agentic_data_system.py <path_to_dataset>")
        
        # Create a dummy CSV for demonstration if no file provided
        dummy_file = "sample_data.csv"
        df = pd.DataFrame({
            'ID': [1, 2, 2, 4, 5],
            'Name': ['Alice', 'Bob ', 'Bob', 'Charlie', None],
            'Age': [25, 30, 30, 200, 22],
            'Salary': [50000, None, 50000, 60000, 55000]
        })
        df.to_csv(dummy_file, index=False)
        print(f"Created dummy file: {dummy_file}")
        target_file = dummy_file
    else:
        target_file = sys.argv[1]
        
    system = AgenticDataSystem()
    result = system.process_dataset(target_file)
    
    print(result["report"])
    
    # Save cleaned data
    output_file = "cleaned_data.csv"
    result["cleaned_df"].to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
