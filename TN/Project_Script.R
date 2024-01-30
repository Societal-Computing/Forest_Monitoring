# Read the CSV file
df <- read.csv("Projects.csv")

# Install httr and jsonlite packages if not already installed
if (!requireNamespace("httr", quietly = TRUE)) {
  install.packages("httr")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

# Load the libraries
library(httr)
library(jsonlite)

# Base URL for the API
base_url <- "https://tree-nation.com/api/projects/"

# Initialize an empty data frame to store the results
results_df <- data.frame()

# Loop through the "id" column in the DataFrame
for (project_id in df$id) {
  # Create the full URL by appending the project_id
  full_url <- paste0(base_url, project_id, "/planting-sites")
  
  # Make the API request
  response <- GET(full_url)
  
  # Check if the response code is 200
  if (status_code(response) == 200) {
    # Parse the JSON response
    json_data <- content(response, "text", encoding = "UTF-8")
    
    # Convert the JSON data to a data frame
    project_data <- fromJSON(json_data, flatten = TRUE)
    
    # Add the project data to the results data frame
    results_df <- rbind(results_df, project_data)
  } else {
    print(paste("Error for project_id", project_id, ": Status Code", status_code(response)))
  }
}

# Print the results data frame
print(results_df)


# Convert dataframe to JSON
json_data <- toJSON(results_df, pretty = TRUE)

# Save JSON to a file
writeLines(json_data, "treenation.json")

