install.packages("jsonlite")
library(jsonlite)



json_file <- "id.json"  # Replace 'path_to_your_json_file.json' with the actual file path
df <- fromJSON(json_file)



# Load required library
library(httr)

# Assuming you have already loaded the 'jsonlite' package

# Function to fetch data from API based on ID
fetch_data <- function(id) {
  url <- paste0("https://restor2-prod-1-api.restor.eco/sites/2/", id)
  response <- GET(url)
  if (http_type(response) == "application/json") {
    content <- content(response, as = "text")
    data <- fromJSON(content)
    return(data)
  } else {
    warning("API did not return JSON data for id ", id)
    return(NULL)
  }
}

# Assuming 'df' is your data frame with 'id' column

# Create an empty list to store data frames
result_list <- list()

# Loop through each ID
for (id in df$id) {
  # Fetch data for each ID
  data <- fetch_data(id)
  
  # If data is not NULL, append it to the result list
  if (!is.null(data)) {
    result_list[[length(result_list) + 1]] <- data
  }
}

# Combine all data frames into one
final_df <- do.call(rbind, result_list)

# Optionally, you can reset row names if needed
rownames(final_df) <- NULL
