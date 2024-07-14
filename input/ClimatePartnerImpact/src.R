# Install and load required packages
library(httr)
library(jsonlite)
library(dplyr)

library(httr)
library(jsonlite)
library(dplyr)
library(tidyr)

# API endpoint
url <- "https://api.explorer.land/v1/public/projects/map?organization_slug=cpi"

# Fetch the data from the API
response <- GET(url)
data <- content(response, "text")
json_data <- fromJSON(data)

# Convert JSON to DataFrame
df <- as.data.frame(json_data$features)

# Unnest the nested columns
df <- df %>%
  unnest_wider(col = geometry, names_sep = "_") %>%
  unnest_wider(col = properties, names_sep = "_")

# Print the first few rows of the dataframe
head(df)


# Remove specified IDs
df <- df %>%
  filter(!(id %in% c("lmedq3qmdby97va4", "e97xby08zejpvo6g")))

# Print the first few rows of the dataframe
head(df)