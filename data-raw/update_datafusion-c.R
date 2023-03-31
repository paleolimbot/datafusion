
curl::curl_download(
  "https://github.com/datafusion-contrib/datafusion-c/releases/download/10.0.0/datafusion-c-10.0.0.zip",
  "data-raw/datafusion-c.zip"
)

unzip("data-raw/datafusion-c.zip", exdir = "data-raw")
