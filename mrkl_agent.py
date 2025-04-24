import os
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")


class MRKLAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.modules = {
            "data_acquisition": self.data_acquisition,
            "data_processing": self.data_processing,
            "visualization": self.visualization_module,
            "knowledge": self.knowledge_module
        }
        self.working_memory = {
            "census_data": None,
            "geo_data": None,
            "merged_data": None,
            "output_path": None
        }
    
    def router(self, task_description: str) -> list:
        """
        Route the task to appropriate modules based on LLM decision

        :param task_description: Description of the task to be executed
        :return: List of module names in the order they should be executed
        """
        print("🧠 MRKL Router: Determining execution plan...")
        
        prompt = f"""
        I need to create a choropleth map of US population density using census data.
        Task details: {task_description}
        
        Which sequence of specialized modules should I use to complete this task?
        Available modules:
        - data_acquisition: Fetches data from external sources like US Census API
        - data_processing: Cleans, transforms, and merges datasets
        - visualization: Creates visual representations and maps
        - knowledge: Provides domain knowledge about US geography, census data interpretation, etc.
        
        For each step, specify exactly one module name from the list above.
        Format your response as a numbered list with only the module names.
        """
        
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=500,
            temperature=0,
            system="You are an expert system designer. Be concise and specific.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the module sequence
        module_sequence = []
        for line in response.content[0].text.split('\n'):
            line = line.strip()
            if line and any(module in line for module in self.modules.keys()):
                for module in self.modules.keys():
                    if module in line:
                        module_sequence.append(module)
                        break
        
        print(f"Execution Plan: {' -> '.join(module_sequence)}")
        return module_sequence
    
    def data_acquisition(self, query: str = None)-> dict:
        """
        Acquiring data from census API and geographic boundaries

        :param query: Optional query to guide data acquisition
        :return: Dictionary with status and message
        """
        print("Data Acquisition Module: Fetching data...")
        
        if query is None:
            query = "Get the latest population data by state from the US Census API and geographic boundaries for US states"
        
        try:
            # Step 1: Fetch Census Population Data
            api_key = os.getenv("CENSUS_API_KEY", "demo_key")  # Default to demo key if not provided
            census_url = f"https://api.census.gov/data/2022/acs/acs5?get=NAME,B01003_001E&for=state:*&key={api_key}"
            
            census_response = requests.get(census_url)
            census_response.raise_for_status()
            
            census_data = census_response.json()
            headers = census_data[0]
            values = census_data[1:]
            
            # Convert to DataFrame
            census_df = pd.DataFrame(values, columns=headers)
            census_df.rename(columns={"B01003_001E": "population"}, inplace=True)
            census_df["population"] = pd.to_numeric(census_df["population"])
            
            self.working_memory["census_data"] = census_df
            
            # Step 2: Fetch Geographic Boundary Data
            geo_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
            
            # Create a temporary directory for downloaded files
            import tempfile
            import zipfile
            
            temp_dir = tempfile.mkdtemp()
            temp_zip = os.path.join(temp_dir, "states.zip")
            
            # Download the shapefile
            geo_response = requests.get(geo_url)
            with open(temp_zip, 'wb') as f:
                f.write(geo_response.content)
            
            # Extract the shapefile
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load the shapefile with GeoPandas
            shapefile_path = os.path.join(temp_dir, "cb_2022_us_state_20m.shp")
            gdf = gpd.read_file(shapefile_path)
            
            self.working_memory["geo_data"] = gdf
            
            return {
                "status": "success",
                "message": f"Successfully acquired census data for {len(census_df)} states and geographic boundaries"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Data acquisition failed: {str(e)}"
            }
    
    def data_processing(self, query: str = None) -> dict:
        """
        Module specialized in data processing and transformation

        :param query: Optional query to guide data processing
        :return: Dictionary with status and message
        """
        print("Processing data")
        
        if query is None:
            query = "Merge population data with geographic boundaries and calculate population density"
        
        try:
            if self.working_memory["census_data"] is None or self.working_memory["geo_data"] is None:
                raise ValueError("Census data or geographic data not available")
            else:
                census_df = self.working_memory["census_data"]
                geo_df = self.working_memory["geo_data"]
            
            # Ask LLM for the data processing approach
            prompt = f"""
            I need to merge US Census population data with geographic boundary data to calculate population density.
            
            Census DataFrame columns: {', '.join(census_df.columns.tolist())}
            First few rows of Census data: {census_df.head(2).to_dict('records')}
            
            Geographic DataFrame columns: {', '.join(geo_df.columns.tolist())}
            
            Provide a precise, step-by-step data processing approach to:
            1. Properly merge these datasets
            2. Calculate population density (population per square kilometer)
            3. Prepare the data for a choropleth map
            
            Use technical, specific instructions that could be converted directly to code.
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system="You are an expert data scientist focusing on geospatial analysis. Be precise and technical.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # We'll follow a standard approach but use the LLM's guidance for reference
            print(f"LLM Processing Guidance: {response.content[0].text[:300]}...")
            
            # Standard processing steps
            # 1. Ensure state codes match for merging
            census_df["state"] = census_df["state"].astype(str)
            geo_df["STATEFP"] = geo_df["STATEFP"].astype(str)
            
            # 2. Merge datasets on state identifier
            merged = geo_df.merge(census_df, left_on="STATEFP", right_on="state")
            
            # 3. Calculate area in square kilometers
            merged["area_sq_km"] = merged.geometry.to_crs("EPSG:3395").area / 10**6
            
            # 4. Calculate population density
            merged["density"] = merged["population"] / merged["area_sq_km"]
            
            # 5. Handle Alaska, Hawaii, and Puerto Rico for continental US map
            continental = merged[~merged["STUSPS"].isin(["AK", "HI", "PR"])]
            
            # 6. Create quantiles for choropleth classification
            continental["density_quantile"] = pd.qcut(
                continental["density"],
                q=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"]
            )
            
            self.working_memory["merged_data"] = continental
            
            return {
                "status": "success",
                "message": f"Successfully processed data for {len(continental)} states. Density range: {continental['density'].min():.2f} to {continental['density'].max():.2f} people/sq km"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Data processing failed: {str(e)}"
            }
    
    def visualization_module(self, query: str = None) -> dict:
        """
        Module specialized in data visualization and mapping

        :param query: Optional query to guide visualization
        :return: Dictionary with status and message
        """
        print("Visualization Module: Creating choropleth map...")
        
        if query is None:
            query = "Create a choropleth map of US population density with appropriate color scheme and styling"
        
        try:
            if self.working_memory["merged_data"] is None:
                raise ValueError("Processed data not available")
            
            data = self.working_memory["merged_data"]
            
            # Ask LLM for visualization recommendations
            prompt = f"""
            I need to create a choropleth map of US population density. The data includes:
            - State boundaries (geometry)
            - Population density (people per square km)
            - Density range: {data['density'].min():.2f} to {data['density'].max():.2f} people/sq km
            
            What visualization parameters would create the most effective choropleth map? Specifically:
            1. What color scheme would be most appropriate?
            2. What classification method should I use (quantiles, equal interval, etc.)?
            3. What styling elements should I include (title, legend, state borders)?
            
            Provide precise, technical recommendations that could be implemented with Matplotlib and GeoPandas.
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=800,
                temperature=0,
                system="You are an expert in data visualization and cartography. Be specific and actionable.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Apply LLM recommendations but with our implementation
            vis_guidance = response.content[0].text
            print(f"LLM Visualization Guidance: {vis_guidance[:300]}...")
            
            # Extract the color scheme from LLM response if possible
            color_scheme = "YlOrRd"  # Yellow-Orange-Red default
            for line in vis_guidance.split("\n"):
                if "color" in line.lower() and "scheme" in line.lower():
                    if "ylord" in line.lower() or "yellow-orange-red" in line.lower():
                        color_scheme = "YlOrRd"
                    elif "blues" in line.lower():
                        color_scheme = "Blues"
                    elif "viridis" in line.lower():
                        color_scheme = "viridis"
                    # Add more color scheme detections as needed
            
            # Create the plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # Plot the choropleth map
            data.plot(
                column="density",
                cmap=color_scheme,
                scheme="quantiles",
                k=5,
                linewidth=0.8,
                ax=ax,
                edgecolor="0.8",
                legend=True,
            )

            # Add legend title separately
            if plt.rcParams["text.usetex"]:
                # For systems with LaTeX
                ax.get_legend().set_title("Population density (per sq km)")
            else:
                # For systems without LaTeX
                ax.get_legend().set_title("Population density (per sq km)")

            # Customize the map
            ax.set_title("US Population Density by State", fontsize=16)
            ax.set_axis_off()
            
            # Customize the map based on LLM suggestions
            ax.set_title("US Population Density by State", fontsize=16)
            ax.set_axis_off()
            
            # Save the map
            output_path = "agent_results/us_population_density_mrkl.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.working_memory["output_path"] = output_path
            
            return {
                "status": "success",
                "message": f"Successfully created population density choropleth map: {output_path}",
                "output_path": output_path
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Visualization failed: {str(e)}"
            }
    
    def knowledge_module(self, query: str = None) -> dict:
        """
        Module specialized in providing domain knowledge

        :param query: Optional query to guide knowledge retrieval
        :return: Dictionary with status and message
        """
        print("Running knowledge module")
        
        prompt = f"""
        I need expert knowledge about US census data and population density mapping.
        
        Query: {query}
        
        Provide a concise, factual response with relevant domain knowledge.
        """
        
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=800,
            temperature=0,
            system="You are an expert demographer and geographer with specialized knowledge of US census data. Provide accurate, concise information.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "status": "success",
            "knowledge": response.content[0].text
        }
    
    def execute_task(self, task_description: str) -> dict:
        """
        Execute the full task using the MRKL approach

        :param task_description: Description of the task to be executed
        :return: Dictionary with task results
        """
        print("Starting MRKL agent")
        
        # Get module sequence from router
        module_sequence = self.router(task_description)
        
        results = []
        for module_name in module_sequence:
            module = self.modules.get(module_name)
            if module:
                # For knowledge module, we need a specific query
                if module_name == "knowledge":
                    if len(results) == 0:  # If it's the first module
                        query = "What is US population density and what patterns should I look for in visualization?"
                    else:  # If it's after some processing
                        query = "How should I interpret population density patterns in the US for an informative visualization?"
                    result = module(query)
                else:
                    result = module()
                
                results.append({
                    "module": module_name,
                    "result": result
                })
                
                # Check for errors
                if result.get("status") == "error":
                    print(f"Error in {module_name}: {result.get('message')}")
                    break
            else:
                print(f"Unknown module: {module_name}")
        
        # Generate final summary
        if self.working_memory.get("output_path"):
            print(f"Task completed successfully. Map saved to: {self.working_memory['output_path']}")
        else:
            print("Task did not complete successfully.")
        
        return {
            "success": self.working_memory.get("output_path") is not None,
            "steps": results,
            "output_path": self.working_memory.get("output_path")
        }

if __name__ == "__main__":
    agent = MRKLAgent()
    task_description = "Create a choropleth map of US population density using the latest census data. The map should visualize population density by state with an appropriate color scheme."
    result = agent.execute_task(task_description)
    
    print("\n=== Task Results ===")
    print(f"Success: {result['success']}")
    
    if result.get("output_path"):
        print(f"Map created at: {result['output_path']}")
    
    # If the Knowledge module was used, display the domain knowledge
    for step in result['steps']:
        if step['module'] == 'knowledge':
            print("\nDomain Knowledge:")
            print(step['result']['knowledge'])
