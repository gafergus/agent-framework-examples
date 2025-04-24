import os
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import anthropic
import tempfile
import zipfile
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")


class ReActAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.state = {
            "census_data": None,
            "geo_data": None,
            "merged_data": None,
            "map_created": False,
            "thoughts": []
        }
        self.tools = {
            "fetch_census_data": self.fetch_census_data,
            "fetch_geo_data": self.fetch_geo_data,
            "process_data": self.process_data,
            "create_map": self.create_map
        }
    
    def think(self, thought: str) -> None:
        """
        Record reasoning steps

        :param thought: A string representing a thought or reasoning step
        """
        self.state["thoughts"].append(thought)
        print(f"Thinking: {thought}")
    
    def fetch_census_data(self):
        """
        Fetch population data from US Census API

        :return: A string indicating success or error message
        """
        self.think("I need to fetch the latest population data from the US Census API")
        
        try:
            api_key = os.getenv("CENSUS_API_KEY", "demo_key")  # Default to the demo key if not provided
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
            
            self.state["census_data"] = df
            return f"Successfully fetched census data for {len(df)} states"
        except Exception as e:
            return f"Error fetching census data: {str(e)}"


    def fetch_geo_data(self) -> str:
        """
        Fetch geographical boundary data for US states

        :return: A string indicating success or error message
        """
        self.think("I need geographic boundary data to create the choropleth map")
        
        try:
            # Using US Census TIGER/Line shapefiles
            url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"

            temp_dir = tempfile.mkdtemp()
            temp_zip = os.path.join(temp_dir, "states.zip")
            
            # Download the shapefile
            response = requests.get(url)
            with open(temp_zip, 'wb') as f:
                f.write(response.content)
            
            # Extract the shapefile
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load the shapefile with GeoPandas
            shapefile_path = os.path.join(temp_dir, "cb_2022_us_state_20m.shp")
            gdf = gpd.read_file(shapefile_path)
            
            self.state["geo_data"] = gdf
            return f"Successfully fetched geographic data with {len(gdf)} boundaries"
        except Exception as e:
            return f"Error fetching geographic data: {str(e)}"
    
    def process_data(self) -> str:
        """
        Process and merge population and geographic data

        :return: A string indicating success or error message
        """
        self.think("I need to merge population data with geographic boundaries and calculate density")
        try:
            if self.state["census_data"] is None or self.state["geo_data"] is None:
                return "Error: Census data or geographic data not available"
            
            census_df = self.state["census_data"]
            geo_df = self.state["geo_data"]
            
            # Ensure state codes match for merging
            census_df["state"] = census_df["state"].astype(str)
            geo_df["STATEFP"] = geo_df["STATEFP"].astype(str)
            # Merge datasets on state identifier
            merged = geo_df.merge(census_df, left_on="STATEFP", right_on="state")
            # Calculate area in square kilometers
            merged["area_sq_km"] = merged.geometry.to_crs("EPSG:3395").area / 10**6
            # Calculate population density
            merged["density"] = merged["population"] / merged["area_sq_km"]
            # Handle Alaska, Hawaii, and Puerto Rico for the continental US map
            continental = merged[~merged["STUSPS"].isin(["AK", "HI", "PR"])]
            self.state["merged_data"] = continental
            return f"Successfully processed data and calculated population density for {len(continental)} states"
        except Exception as e:
            return f"Error processing data: {str(e)}"
    
    def create_map(self) -> str:
        """
        Create the choropleth map visualizing population density

        :return: A string indicating success or error message
        """
        self.think("I need to create a visually informative choropleth map with appropriate color scheme")
        
        try:
            if self.state["merged_data"] is None:
                return "Error: Processed data not available"
            
            data = self.state["merged_data"]
            
            # Create the plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # Plot the choropleth map
            data.plot(
                column="density",
                cmap="YlOrRd",  # Yellow-Orange-Red color scheme for population density
                linewidth=0.8,
                ax=ax,
                edgecolor="0.8",
                legend=True,
                legend_kwds={"label": "Population density (per sq km)"}
            )
            
            # Customize the map
            ax.set_title("US Population Density by State", fontsize=16)
            ax.set_axis_off()
            
            # Save the map
            plt.savefig("us_population_density_react.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            self.state["map_created"] = True
            return "Successfully created population density choropleth map: us_population_density_react.png"
        except Exception as e:
            return f"Error creating map: {str(e)}"
    
    def run_task(self) -> dict:
        """
        Execute the full task using ReAct approach

        :return: A dictionary with success status, steps taken, and thoughts
        """
        # Start with a high-level plan
        self.think("To create a US population density choropleth map, I need to: 1) Get census data, 2) Get geographic data, 3) Process and merge data, 4) Create map")
        
        # Execute the plan step by step
        results = [self.tools["fetch_census_data"], self.tools["fetch_geo_data"], self.tools["process_data"],
                   self.tools["create_map"]]
        
        # Step 1: Fetch census data
        results[0]()
        # Step 2: Fetch geographic data
        results[1]()
        # Step 3: Process the data
        results[2]()
        # Step 4: Create the map
        results[3]()

        # Generate final reflection
        if self.state["map_created"]:
            self.think("Task completed successfully. The map shows population density variations across US states.")
        else:
            self.think("Task encountered issues. Need to diagnose what went wrong in the process.")
        
        return {
            "success": self.state["map_created"],
            "steps": results,
            "thoughts": self.state["thoughts"]
        }
    
    def run_with_llm_guidance(self) -> dict:
        """
        Run the task with LLM guidance for each step

        :return: A dictionary with success status, steps taken, and thoughts
        """
        print("Starting ReAct agent with LLM guidance...")
        
        # Initial prompt to get the LLM to guide the process
        prompt = """
        You are helping create a choropleth map of US population density using census data.
        Here's the current state of the task:
        - Census data: {census_data_status}
        - Geographic data: {geo_data_status}
        - Data processing: {processing_status}
        - Map creation: {map_status}
        
        What should be the next step? Provide reasoning and then specify ONE action to take from these options:
        1. fetch_census_data
        2. fetch_geo_data
        3. process_data
        4. create_map
        5. task_complete
        
        Format your response as:
        Reasoning: your reasoning here
        Action: action_name
        """
        
        results = []
        max_steps = 10  # Prevent infinite loops
        steps_taken = 0
        
        while steps_taken < max_steps:
            # Update status
            census_status = "Available" if self.state["census_data"] is not None else "Not fetched"
            geo_status = "Available" if self.state["geo_data"] is not None else "Not fetched"
            processing_status = "Completed" if self.state["merged_data"] is not None else "Not processed"
            map_status = "Created" if self.state["map_created"] else "Not created"
            
            # Get LLM decision
            current_prompt = prompt.format(
                census_data_status=census_status,
                geo_data_status=geo_status,
                processing_status=processing_status,
                map_status=map_status
            )
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are an expert data scientist helping to create a US population density map. Be concise and decisive.",
                messages=[
                    {"role": "user", "content": current_prompt}
                ]
            )
            
            llm_text = response.content[0].text
            
            # Parse LLM response
            reasoning = ""
            action = ""
            for line in llm_text.split("\n"):
                if line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
                elif line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
            
            # Record LLM's reasoning
            self.think(reasoning)
            
            # Execute the action
            if action == "task_complete":
                break
            elif action in self.tools:
                result = self.tools[action]()
                results.append(result)
            else:
                results.append(f"Unknown action: {action}")
            
            steps_taken += 1
            
            # Check if we've completed the task
            if self.state["map_created"]:
                self.think("Map has been created successfully. Task complete.")
                break
        
        return {
            "success": self.state["map_created"],
            "steps": results,
            "thoughts": self.state["thoughts"]
        }

if __name__ == "__main__":
    agent = ReActAgent()
    result = agent.run_with_llm_guidance()
    
    print("\n=== Task Results ===")
    print(f"Success: {result['success']}")
    print("\nSteps Taken:")
    for i, step in enumerate(result['steps']):
        print(f"{i+1}. {step}")
    
    print("\nMap created at: us_population_density_react.png")
