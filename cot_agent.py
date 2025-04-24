import os
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import anthropic
import tempfile
import zipfile
import json
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any, List

load_dotenv("../.env")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable with your API key.")


class ChainOfThoughtAgent:
    """
    Implementation of a Chain-of-Thought agent for creating a choropleth map of US population density
    using explicit reasoning steps.
    """

    def __init__(self, log_to_file: bool = True, debug: bool = False):
        """
        Initialize the Chain-of-Thought agent.

        :param log_to_file: Whether to save reasoning steps to a log file
        """
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.reasoning_chain: List[Dict[str, str]] = []
        self.log_to_file = log_to_file
        self.log_file: Optional[str] = None

        # Fix: Using type annotations to help the type checker
        self.data: Dict[str, Any] = {
            "census_data": None,  # Will be pd.DataFrame
            "geo_data": None,  # Will be gpd.GeoDataFrame
            "merged_data": None,  # Will be gpd.GeoDataFrame
            "output_path": None  # Will be str
        }

        if self.log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("logs", exist_ok=True)
            self.log_file = f"logs/cot_reasoning_{timestamp}.json"

        self.debug = debug

    def add_reasoning(self, step_name: str, reasoning: str):
        """
        Add a reasoning step to the chain and log it.

        :param step_name: Name of the current step
        :param reasoning: Reasoning text explaining the approach
        """
        step_entry = {
            "step": step_name,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

        self.reasoning_chain.append(step_entry)
        print(f"\n{'=' * 20} STEP: {step_name} {'=' * 20}")
        print(f"{reasoning[:300]}..." if len(reasoning) > 300 else reasoning)
        if self.log_to_file and self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.reasoning_chain, f, indent=2)

    def decompose_task(self, task_description: str) -> list:
        """
        Use LLM to decompose the task into logical steps with detailed reasoning.

        :param task_description: Description of the overall task
        :return: A list of detailed steps with explanations
        """
        self.add_reasoning("Initial Task", f"Original task: {task_description}")

        prompt = f"""
        Task: {task_description}

        I need to complete this task using a chain-of-thought approach, where I break down the problem
        into clear logical steps and reason through each step explicitly before executing it.

        Please help me decompose this task into 4-6 well-defined steps. For each step:

        1. Provide a clear title describing what needs to be done
        2. Explain the detailed reasoning for why this step is necessary
        3. Describe what data or information will be needed for this step
        4. Outline any potential challenges or considerations for this step
        5. Explain what the expected output from this step should be

        Be specific and detailed in your reasoning. Focus on creating a logical progression where each
        step builds on the previous ones, ultimately leading to completing the entire task.
        """
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0,
            system="You are an expert data scientist who specializes in geospatial analysis. You break down complex problems into logical steps with detailed reasoning.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Add the decomposition reasoning to our chain
        decomposition = response.content[0].text
        self.add_reasoning("Task Decomposition", decomposition)

        # Print the decomposition for debugging
        if self.debug:
            print(f"\n{'=' * 20} Task Decomposition {'=' * 20}")
            print(f"{decomposition}")
            print("\n")
            print(f"\n{'=' * 20} SELF.ADD_REASONING {'=' * 20}")
            print(f"{self.reasoning_chain}")


        # Extract steps from the decomposition
        steps = self._parse_decomposition(decomposition)
        return steps

    @staticmethod
    def _parse_decomposition(decomposition: str) -> List[Dict[str, str]]:
        """
        Parse the LLM's task decomposition into structured steps.

        :param decomposition: Raw text from LLM with task decomposition
        :return: List of dictionaries with structured step information
        """
        lines = decomposition.split('\n')
        steps: List[Dict[str, str]] = []
        current_step: Optional[Dict[str, str]] = None
        current_section: Optional[str] = None

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check for step headers in Markdown format: ## Step N: or ## Step N: Title
            if line_str.startswith('## Step '):
                if current_step is not None:
                    steps.append(current_step)

                # Extract the title from the header
                parts = line_str.split(':', 1)
                if len(parts) > 1:
                    step_title = parts[1].strip()
                else:
                    # If there's no colon, just use the whole line as the title
                    step_title = line_str.replace('## Step ', '').strip()

                current_step = {
                    "title": step_title,
                    "reasoning": "",
                    "data_needed": "",
                    "challenges": "",
                    "expected_output": ""
                }
                current_section = "reasoning"

            # Check for section headers within steps (marked with ** in Markdown)
            elif current_step is not None and line_str.startswith('**') and line_str.endswith('**'):
                section_name = line_str.strip('**').lower()
                if "reasoning" in section_name:
                    current_section = "reasoning"
                elif "data needed" in section_name:
                    current_section = "data_needed"
                elif "potential challenges" in section_name:
                    current_section = "challenges"
                elif "expected output" in section_name:
                    current_section = "expected_output"

            # Add content to the current section, skipping section headers which we've already processed
            elif current_step is not None and current_section is not None and not line_str.startswith('**'):
                # Skip lines that are likely headers (##) or bullet points if they don't contain content
                if line_str.startswith('#') or line_str.startswith('-'):
                    # Only add bullet points which likely contain actual content
                    if line_str.startswith('- ') and len(line_str) > 2:
                        current_step[current_section] += line_str + " "
                else:
                    # Regular content lines
                    current_step[current_section] += line_str + " "

        # Add the last step
        if current_step is not None:
            steps.append(current_step)

        return steps

    def reason_through_step(self, step_info: dict) -> str:
        """
        Generate detailed reasoning for how to approach a specific step.

        :param step_info: Dictionary with step information
        :return: Detailed reasoning for the step
        """
        step_title = step_info["title"]
        step_reasoning = step_info.get("reasoning", "")

        prompt = f"""
        I'm working on this step in my task: "{step_title}"

        Background reasoning for this step: {step_reasoning}

        Current state of my data:
        - Census data: {"Available" if self.data["census_data"] is not None else "Not yet acquired"}
        - Geographic data: {"Available" if self.data["geo_data"] is not None else "Not yet acquired"}
        - Merged data: {"Available" if self.data["merged_data"] is not None else "Not yet created"}

        I need to reason through exactly how to implement this step using Python code.
        Think step-by-step about:

        1. What specific Python libraries and functions would be most appropriate
        2. The exact sequence of operations needed
        3. Potential edge cases or challenges I should handle
        4. How to validate that the step was completed successfully

        Be specific and detailed in your reasoning. Focus on the "how" rather than the "what" since I already 
        know what needs to be done. I need your help to reason through the implementation details.
        """

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            temperature=0,
            system="You are an expert Python programmer specializing in data analysis and geospatial visualization. Think through implementation details carefully and thoroughly.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        implementation_reasoning = response.content[0].text
        self.add_reasoning(f"Implementation: {step_title}", implementation_reasoning)
        return implementation_reasoning

    def execute_fetch_census_data(self) -> dict:
        """
        Execute the step to fetch population data from US Census API.
        Includes error handling and validation.

        :return: Dictionary with execution status and results
        """
        try:
            self.add_reasoning("Executing: Fetch Census Data",
                               "Fetching population data from the US Census API using the ACS 5-year estimates.")


            # Using Census API to get population data by state
            api_key = os.getenv("CENSUS_API_KEY", "demo_key")
            url = f"https://api.census.gov/data/2022/acs/acs5?get=NAME,B01003_001E&for=state:*&key={api_key}"

            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            headers = data[0]
            values = data[1:]

            # Convert to DataFrame
            df = pd.DataFrame(values, columns=headers)
            df.rename(columns={"B01003_001E": "population"}, inplace=True)
            df["population"] = pd.to_numeric(df["population"])

            # Validate the data
            if len(df) < 50:
                raise ValueError(f"Expected at least 50 states, but only got {len(df)}")

            # Add some validation of the data
            self.add_reasoning("Validation: Census Data",
                               f"Successfully fetched data for {len(df)} states. "
                               f"Population range: {df['population'].min():,} to {df['population'].max():,} people. "
                               f"Total US population (sum of states): {df['population'].sum():,}")

            # Fix: Type checking issue - explicitly cast data if needed
            self.data["census_data"] = df

            return {
                "status": "success",
                "message": f"Successfully fetched census data for {len(df)} states",
                "data_shape": df.shape,
                "has_required_columns": all(col in df.columns for col in ["NAME", "population", "state"])
            }
        except Exception as e:
            error_msg = f"Error fetching census data: {str(e)}"
            self.add_reasoning("Error: Fetch Census Data", error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def execute_fetch_geo_data(self) -> dict:
        """
        Execute the step to fetch geographical boundary data for US states.
        Includes error handling and validation.

        :return: Dictionary with execution status and results
        """
        try:
            self.add_reasoning("Executing: Fetch Geographic Data",
                               "Fetching US state boundary shapefiles from Census TIGER/Line.")

            # Using US Census TIGER/Line shapefiles
            url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"

            # Create a temporary directory for downloaded files
            temp_dir = tempfile.mkdtemp()
            temp_zip = os.path.join(temp_dir, "states.zip")

            # Download the shapefile
            response = requests.get(url)
            with open(temp_zip, 'wb') as f:
                f.write(response.content)

            # Extract the shapefiolor
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Load the shapefile with GeoPandas
            shapefile_path = os.path.join(temp_dir, "cb_2022_us_state_20m.shp")
            gdf = gpd.read_file(shapefile_path)

            # Validate the data
            if len(gdf) < 50:
                raise ValueError(f"Expected at least 50 states, but only got {len(gdf)}")

            # Add some validation of the data
            self.add_reasoning("Validation: Geographic Data",
                               f"Successfully loaded geographic data for {len(gdf)} states/territories. "
                               f"CRS: {gdf.crs}. "
                               f"Columns available: {', '.join(gdf.columns)}")

            # Fix: Type checking issue
            self.data["geo_data"] = gdf

            return {
                "status": "success",
                "message": f"Successfully fetched geographic data with {len(gdf)} boundaries",
                "data_shape": gdf.shape,
                "crs": str(gdf.crs),
                "has_required_columns": all(col in gdf.columns for col in ["STATEFP", "NAME", "geometry"])
            }
        except Exception as e:
            error_msg = f"Error fetching geographic data: {str(e)}"
            self.add_reasoning("Error: Fetch Geographic Data", error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def execute_process_data(self) -> dict:
        """
        Execute the step to process and merge population and geographic data.
        Calculates population density and prepares data for visualization.

        :return: Dictionary with execution status and results
        """
        try:
            self.add_reasoning("Executing: Process Data",
                               "Merging census population data with geographic boundaries and calculating population density.")

            # Fix: Check for None properly
            census_df = self.data["census_data"]
            geo_df = self.data["geo_data"]

            if census_df is None or geo_df is None:
                raise ValueError("Census data or geographic data not available. "
                                 "Please execute the data fetching steps first.")

            # Ensure state codes match for merging
            census_df["state"] = census_df["state"].astype(str)
            geo_df["STATEFP"] = geo_df["STATEFP"].astype(str)

            # Merge datasets on state identifier
            merged = geo_df.merge(census_df, left_on="STATEFP", right_on="state")

            # Check that we didn't lose data in the merge
            if len(merged) < min(len(census_df), len([x for x in geo_df["STATEFP"] if x in census_df["state"].values])):
                self.add_reasoning("Warning: Data Processing",
                                   f"Possible data loss during merge. Started with {len(census_df)} census records and "
                                   f"{len(geo_df)} geographic records, but ended with only {len(merged)} merged records.")

            # Calculate area in square kilometers
            merged["area_sq_km"] = merged.geometry.to_crs("EPSG:3395").area / 10 ** 6

            # Calculate population density
            merged["density"] = merged["population"] / merged["area_sq_km"]

            # Add a log-transformed density for better visualization
            merged["log_density"] = np.log10(merged["density"])

            # Handle Alaska, Hawaii, and Puerto Rico for the continental US map
            continental = merged[~merged["STUSPS"].isin(["AK", "HI", "PR"])]

            # Create classification for legend
            breaks = [0, 10, 25, 50, 100, 200, 1000]
            labels = ['< 10', '10-25', '25-50', '50-100', '100-200', '> 200']
            continental['density_category'] = pd.cut(
                continental['density'],
                bins=breaks,
                labels=labels,
                include_lowest=True
            )

            # Fix: Type checking issue
            self.data["merged_data"] = continental

            # Add some validation of the processed data
            self.add_reasoning("Validation: Processed Data",
                               f"Successfully processed data for {len(continental)} continental US states. "
                               f"Density range: {continental['density'].min():.2f} to {continental['density'].max():.2f} people/sq km. "
                               f"Median density: {continental['density'].median():.2f} people/sq km.")

            # Create a summary table for reasoning
            density_summary = continental.groupby('density_category', observed=False)['NAME_x'].count().reset_index()
            density_summary.columns = ['Density Category', 'Number of States']
            summary_str = "Density distribution across states:\n" + density_summary.to_string(index=False)
            self.add_reasoning("Analysis: Population Density", summary_str)

            return {
                "status": "success",
                "message": f"Successfully processed data and calculated population density for {len(continental)} states",
                "density_range": (continental['density'].min(), continental['density'].max()),
                "density_distribution": density_summary.to_dict()
            }
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.add_reasoning("Error: Process Data", error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def execute_create_map(self):
        """
        Execute the step to create the choropleth map visualizing population density.
        Includes consulting the LLM for optimal visualization parameters.

        :return: Dictionary with execution status and results
        """
        try:
            self.add_reasoning("Executing: Create Visualization",
                               "Creating a choropleth map to visualize US population density.")

            # Fix: Check for None properly and handle it
            data = self.data["merged_data"]
            if data is None:
                raise ValueError("Processed data not available. Please execute the data processing step first.")

            # Get LLM advice on visualization parameters
            prompt = f"""
            I need to create an effective choropleth map of US population density with these characteristics:
            - Data range: {data['density'].min():.2f} to {data['density'].max():.2f} people per sq km
            - Median: {data['density'].median():.2f} people per sq km
            - Mean: {data['density'].mean():.2f} people per sq km
            - Most states have density below 100 people per sq km, but a few are much higher

            What visualization parameters would create the most effective and informative map?
            Specifically recommend:
            1. Most appropriate color scheme (specific matplotlib colormap)
            2. Classification method (quantiles, natural breaks, or equal interval)
            3. Number of classes/bins
            4. Whether to use a linear or logarithmic scale
            5. Any specific styling recommendations for clearer information display

            Explain the reasoning behind each recommendation.
            """

            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are an expert cartographer specializing in thematic mapping and visualization design. Provide specific, actionable recommendations.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            viz_recommendations = response.content[0].text
            self.add_reasoning("Visualization Design", viz_recommendations)

            # Parse visualization recommendations
            # Default values
            color_scheme = "YlOrRd"
            classification = "quantiles"
            num_classes = 5
            log_scale = True

            # Try to extract recommendations from LLM response
            for line in viz_recommendations.split('\n'):
                if "color" in line.lower() and "scheme" in line.lower():
                    if "ylord" in line.lower() or "yellow-orange-red" in line.lower():
                        color_scheme = "YlOrRd"
                    elif "ylorbr" in line.lower() or "yellow-orange-brown" in line.lower():
                        color_scheme = "YlOrBr"
                    elif "rdylgn" in line.lower() or "red-yellow-green" in line.lower():
                        color_scheme = "RdYlGn_r"
                    elif "viridis" in line.lower():
                        color_scheme = "viridis"
                    elif "plasma" in line.lower():
                        color_scheme = "plasma"

                if "classification" in line.lower() or "method" in line.lower():
                    if "natural" in line.lower() and "breaks" in line.lower():
                        classification = "natural_breaks"
                    elif "equal" in line.lower() and "interval" in line.lower():
                        classification = "equal_interval"
                    elif "quantile" in line.lower():
                        classification = "quantiles"

                if "log" in line.lower() and "scale" in line.lower():
                    log_scale = "recommend" in line.lower() or "use" in line.lower() or "better" in line.lower()

                if "class" in line.lower() or "bin" in line.lower():
                    for i in range(3, 10):
                        if str(i) in line:
                            num_classes = i
                            break

            # Create the figure and axes
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            # Determine the column to plot
            plot_column = "log_density" if log_scale else "density"

            # Plot the choropleth map
            if classification == "equal_interval":
                # For equal interval, we need to specify the scheme differently
                data.plot(
                    column=plot_column,
                    cmap=color_scheme,
                    linewidth=0.8,
                    ax=ax,
                    edgecolor="0.8",
                    legend=True,
                    scheme="equal_interval",
                    k=num_classes,
                    legend_kwds={
                        "title": "Population density (people per sq km)",
                        "loc": "lower right"
                    }
                )
            else:
                data.plot(
                    column=plot_column,
                    cmap=color_scheme,
                    scheme=classification,
                    k=num_classes,
                    linewidth=0.8,
                    ax=ax,
                    edgecolor="0.8",
                    legend=True,
                    legend_kwds={
                        "title": "Population density (people per sq km)" + (" (log scale)" if log_scale else ""),
                        "loc": "lower right"
                    }
                )

            # Customize the map
            ax.set_title("US Population Density by State", fontsize=16)
            ax.set_axis_off()

            # Add a note about the data source
            fig.text(0.1, 0.01, "Data source: US Census Bureau, ACS 5-Year Estimates (2022)",
                     fontsize=10, color='gray')

            # Save the map
            output_path = "agent_results/us_population_density_cot.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Fix: Type checking issue
            self.data["output_path"] = output_path

            self.add_reasoning("Map Creation Complete",
                               f"Successfully created population density choropleth map: {output_path}\n"
                               f"Used {color_scheme} color scheme with {classification} classification method and {num_classes} classes.\n"
                               f"{'Applied logarithmic scale for better visualization of the wide range of values.' if log_scale else ''}")

            return {
                "status": "success",
                "message": f"Successfully created population density choropleth map: {output_path}",
                "output_path": output_path,
                "visualization_params": {
                    "color_scheme": color_scheme,
                    "classification": classification,
                    "num_classes": num_classes,
                    "log_scale": log_scale
                }
            }
        except Exception as e:
            error_msg = f"Error creating map: {str(e)}"
            self.add_reasoning("Error: Create Map", error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def reflect_on_results(self) -> dict:
        """
        Final reflection on the entire process and results.

        :return: Dictionary with reflection and overall status
        """
        # Fix: Check for None properly
        output_path = self.data["output_path"]
        if output_path is None:
            self.add_reasoning("Final Reflection",
                               "The task was not completed successfully. No output map was generated.")
            return {
                "status": "error",
                "message": "Task was not completed successfully"
            }

        # Get LLM to reflect on the process
        prompt = f"""
        I've completed the task of creating a US population density choropleth map using a chain-of-thought approach.

        Here's a summary of what was done:
        1. Fetched population data from the US Census API
        2. Fetched geographic boundary data from Census TIGER/Line
        3. Processed and merged the data to calculate population density
        4. Created a choropleth map visualization using {self.data.get("visualization_params", {}).get("color_scheme", "unknown")} color scheme

        The map has been saved to {output_path}.

        Please provide a reflection on:
        1. The strengths and limitations of the approach taken
        2. Potential improvements or extensions to the analysis
        3. Insights about US population density patterns that could be highlighted
        4. How this approach compares to other frameworks for LLM-guided data analysis
        """

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            temperature=0,
            system="You are an expert data scientist with deep knowledge of geospatial analysis and LLM-guided workflows. Provide thoughtful reflection on completed work.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        reflection = response.content[0].text
        self.add_reasoning("Final Reflection", reflection)

        return {
            "status": "success",
            "message": "Task completed successfully with final reflection",
            "output_path": output_path,
            "reflection": reflection
        }

    def execute_step(self, step_title: str, step_index: int) -> dict:
        """
        Execute a single step based on its title and position in the sequence.

        :param step_title: Title of the step to execute
        :param step_index: Index of the step in the sequence (0-based)
        :return: Result of the step execution
        """
        # Handle the step based on its position in the sequence
        if step_index == 0:  # First step: Get census data
            return self.execute_fetch_census_data()
        elif step_index == 1:  # Second step: Get geographic data
            return self.execute_fetch_geo_data()
        elif step_index == 2:  # Third step: Process and merge data
            return self.execute_process_data()
        elif step_index >= 3:  # Remaining steps: Visualization and reflection
            # Check dependencies before executing
            if self.data["merged_data"] is None:
                return {
                    "status": "error",
                    "message": "Cannot create visualization without processed data. Steps must be executed in order."
                }
            return self.execute_create_map()
        else:
            self.add_reasoning("Error: Unknown Step", f"Could not determine how to execute step: {step_title}")
            return {
                "status": "error",
                "message": f"Unknown step: {step_title}"
            }

    def run_task(self, task_description: str) -> dict:
        """
        Execute the full task using the Chain-of-Thought approach.

        :param task_description: Description of the overall task
        :return: Dictionary with overall execution results
        """
        print(f"\n{'=' * 30} STARTING CHAIN-OF-THOUGHT AGENT {'=' * 30}")
        print(f"Task: {task_description}")

        # Step 1: Decompose the task into logical steps
        steps = self.decompose_task(task_description)

        #print(f"\n{'=' * 30} DECOMPOSED STEPS {'=' * 30}")
        #print(f"Steps: {steps}")

        # Step 2: Process each step with explicit reasoning
        results = []
        for i, step in enumerate(steps):
            print(f"\n{'=' * 10} STEP {i + 1}/{len(steps)}: {step['title']} {'=' * 10}")
            print(f"Task: {step['title']}")

            # First, reason through how to implement the step
            self.reason_through_step(step)

            # Then execute the step - passing the index
            result = self.execute_step(step['title'], i)

        # Step 3: Final reflection on the entire process
        final_result = self.reflect_on_results()

        print(f"\n{'=' * 30} CHAIN-OF-THOUGHT EXECUTION COMPLETE {'=' * 30}")
        output_path = self.data["output_path"]
        if output_path:
            print(f"Output map created at: {output_path}")
            print(f"Detailed reasoning log saved to: {self.log_file}")

        return {
            "success": output_path is not None,
            "steps": results,
            "reasoning_chain": self.reasoning_chain,
            "output_path": output_path,
            "final_reflection": final_result.get("reflection")
        }


if __name__ == "__main__":
    # Define the task
    task_description = ("Create a choropleth map of US population density using census data. The map should visualize "
                        "population density by state, using an appropriate color scheme and classification method.")

    # Initialize and run the agent
    agent = ChainOfThoughtAgent()
    result = agent.run_task(task_description)

    # Print the overall result
    print("\n=== Task Execution Summary ===")
    print(f"Success: {result['success']}")

    if result.get("output_path"):
        print(f"Map created at: {result['output_path']}")

    # Print a short excerpt of the final reflection
    if result.get("final_reflection"):
        print("\nFinal Reflection Excerpt:")
        print(result['final_reflection'][:300] + "..." if len(result['final_reflection']) > 300 else result[
            'final_reflection'])
