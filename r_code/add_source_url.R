# Add original PDF URL column

# Packages ----
pkgs <- c("readr","dplyr","stringr","tibble","writexl")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
lapply(pkgs, library, character.only = TRUE)

if (!requireNamespace("httr", quietly = TRUE)) {
  try(install.packages("httr", repos = "https://cloud.r-project.org"), silent = TRUE)
}

# Paths
INPUT_CSV  <- "C:/Users/rakes/Music/cities/CITES_Data/master data/cites.cops.csv"
OUTPUT_CSV <- "C:/Users/rakes/Music/cities/CITES_Data/master data/cites.cops.with_URL.csv"
OUTPUT_XLSX<- "C:/Users/rakes/Music/cities/CITES_Data/master data/cites.cops.with_URL.xlsx"

#Read (do NOT rename/clean columns; keep data as-is) 
df <- readr::read_csv(
  INPUT_CSV,
  show_col_types = FALSE,
  name_repair = "unique"   
)

detect_col <- function(.df, candidates) {
  nms <- names(.df)
  low <- tolower(nms)
  # exact match (case-insensitive)
  for (c in candidates) {
    hit <- which(low == tolower(c))
    if (length(hit)) return(nms[hit[1]])
  }
  # contains match (case-insensitive)
  for (c in candidates) {
    hit <- which(stringr::str_detect(low, fixed(tolower(c))))
    if (length(hit)) return(nms[hit[1]])
  }
  return(NULL)
}

derive_cop_num <- function(.df) {
  # If cop_num already exists, keep it (as character)
  cop_num_col <- detect_col(.df, c("cop_num","copnum","cop number"))
  if (!is.null(cop_num_col)) {
    .df[[cop_num_col]] <- as.character(.df[[cop_num_col]])
    return(list(df=.df, col=cop_num_col))
  }

  cop_col <- detect_col(.df, c("cop","COP","CoP"))
  if (!is.null(cop_col)) {
    .df$cop_num <- as.character(readr::parse_number(as.character(.df[[cop_col]])))
    return(list(df=.df, col="cop_num"))
  }

  # Else try Meeting column like "CoP12"
  meeting_col <- detect_col(.df, c("meeting","Meeting"))
  if (!is.null(meeting_col)) {
    .df$cop_num <- stringr::str_extract(as.character(.df[[meeting_col]]), "\\d+")
    return(list(df=.df, col="cop_num"))
  }

  stop("Could not derive cop_num: no COP/CoP or Meeting/meeting column found.")
}

is_http_url <- function(x) {
  !is.na(x) & stringr::str_detect(x, "^https?://")
}

# Ensure cop_num exists
tmp <- derive_cop_num(df)
df <- tmp$df
COPNUM_COL <- tmp$col

# Identify Status column
STATUS_COL <- detect_col(df, c("status","Status","group","Group"))
SRC_COL <- detect_col(df, c("source_pdf_url","source url","source_pdf","pdf","pdf_path","sourcepath","source_pdf_path"))

# Keep original source column untouched; create helper columns safely
if (!is.null(SRC_COL)) {
  df <- df %>%
    mutate(
      source_pdf_raw  = as.character(.data[[SRC_COL]]),
      source_pdf_http = ifelse(is_http_url(source_pdf_raw), source_pdf_raw, NA_character_),
      source_pdf_file = ifelse(!is.na(source_pdf_raw) & source_pdf_raw != "",
                               basename(source_pdf_raw), NA_character_)
    )
} else {
  df <- df %>%
    mutate(
      source_pdf_raw  = NA_character_,
      source_pdf_http = NA_character_,
      source_pdf_file = NA_character_
    )
}

url_map <- tibble::tribble(
  ~cop_num, ~status,      ~URL,

  # CoP1
  "1",     "parties",    NA_character_,
  "1",     "observers",  NA_character_,

  # CoP2–7
  "2",     "parties",    "https://cites.org/sites/default/files/eng/cop/02/E02-List-of-participants-Parties.pdf",
  "2",     "observers",  "https://cites.org/sites/default/files/eng/cop/02/E02-List-of-participants-Observers.pdf",

  "3",     "parties",    "https://cites.org/sites/default/files/eng/cop/03/E03-List-of-Participants-Parties.pdf",
  "3",     "observers",  "https://cites.org/sites/default/files/eng/cop/03/E03-List-of-Participants-Observers.pdf",

  "4",     "parties",    "https://cites.org/sites/default/files/eng/cop/04/E04-List-of-Participants-Parties.pdf",
  "4",     "observers",  "https://cites.org/sites/default/files/eng/cop/04/E04-List-of-Participants-Observers.pdf",

  "5",     "parties",    "https://cites.org/sites/default/files/eng/cop/05/E05-List-of-Participants-Parties.pdf",
  "5",     "observers",  "https://cites.org/sites/default/files/eng/cop/05/E05-List-of-Participants-Observers.pdf",

  "6",     "parties",    "https://cites.org/sites/default/files/eng/cop/06/E06-List-of-Participants-Parties.pdf",
  "6",     "observers",  "https://cites.org/sites/default/files/eng/cop/06/E06-List-of-Participants-Observers.pdf",

  "7",     "parties",    "https://cites.org/sites/default/files/eng/cop/07/E07-List-of-Participants-Parties.pdf",
  "7",     "observers",  "https://cites.org/sites/default/files/eng/cop/07/E07-List-of-Participants-Observers.pdf",

  # CoP8–11 (single PDF covers both)
  "8",     "parties",    "https://cites.org/sites/default/files/eng/cop/08/E-Participants.pdf",
  "8",     "observers",  "https://cites.org/sites/default/files/eng/cop/08/E-Participants.pdf",

  "9",     "parties",    "https://cites.org/sites/default/files/eng/cop/09/E9-participants.pdf",
  "9",     "observers",  "https://cites.org/sites/default/files/eng/cop/09/E9-participants.pdf",

  "10",    "parties",    "https://cites.org/sites/default/files/eng/cop/10/E10-participants.pdf",
  "10",    "observers",  "https://cites.org/sites/default/files/eng/cop/10/E10-participants.pdf",

  "11",    "parties",    "https://cites.org/sites/default/files/eng/cop/11/other/list_participants.pdf",
  "11",    "observers",  "https://cites.org/sites/default/files/eng/cop/11/other/list_participants.pdf",

  # CoP12–15 (separate PDFs)
  "12",    "parties",    "https://cites.org/sites/default/files/common/cop/12/participants_party.pdf",
  "12",    "observers",  "https://cites.org/sites/default/files/common/cop/12/participants_observer.pdf",

  "13",    "parties",    "https://cites.org/sites/default/files/common/cop/13/participants_party.pdf",
  "13",    "observers",  "https://cites.org/sites/default/files/common/cop/13/participants_observer.pdf",

  "14",    "parties",    "https://cites.org/sites/default/files/common/cop/14/list_party.pdf",
  "14",    "observers",  "https://cites.org/sites/default/files/common/cop/14/list_observer.pdf",

  "15",    "parties",    "https://cites.org/sites/default/files/common/cop/15/cop15_participants_parties.pdf",
  "15",    "observers",  "https://cites.org/sites/default/files/common/cop/15/cop15_participants_observers.pdf",

  # CoP16–20 (single PDF covers both)
  "16",    "parties",    "https://cites.org/sites/default/files/eng/cop/16/OfficialListofParticipants.pdf",
  "16",    "observers",  "https://cites.org/sites/default/files/eng/cop/16/OfficialListofParticipants.pdf",

  "17",    "parties",    "https://cites.org/sites/default/files/eng/cop/17/FinalCoP17ParticipantList.pdf",
  "17",    "observers",  "https://cites.org/sites/default/files/eng/cop/17/FinalCoP17ParticipantList.pdf",

  "18",    "parties",    "https://cites.org/sites/default/files/eng/cop/18/CoP18%20participants%20final.pdf",
  "18",    "observers",  "https://cites.org/sites/default/files/eng/cop/18/CoP18%20participants%20final.pdf",

  "19",    "parties",    "https://cites.org/sites/default/files/eng/cop/19/CoP19-LoP-final.pdf",
  "19",    "observers",  "https://cites.org/sites/default/files/eng/cop/19/CoP19-LoP-final.pdf",

  "20",    "parties",    "https://cites.org/sites/default/files/eng/cop/20/CoP20-LoP-final.pdf",
  "20",    "observers",  "https://cites.org/sites/default/files/eng/cop/20/CoP20-LoP-final.pdf"
)

# normalize url_map fields
url_map <- url_map %>%
  mutate(
    cop_num = as.character(cop_num),
    status  = ifelse(is.na(status), NA_character_, tolower(as.character(status))),
    URL     = dplyr::na_if(as.character(URL), "")
  )


df2 <- df

# normalize Status for join if we have it
if (!is.null(STATUS_COL)) {
  df2 <- df2 %>% mutate(status_join = tolower(as.character(.data[[STATUS_COL]])))
} else {
  df2 <- df2 %>% mutate(status_join = NA_character_)
}

# Join logic:
# If url_map has any non-NA status values AND df has Status, join on (cop_num, status)
# Else join only on cop_num
if (!is.null(STATUS_COL) && any(!is.na(url_map$status))) {
  df2 <- df2 %>%
    left_join(url_map, by = c(setNames("cop_num", COPNUM_COL), "status_join" = "status"))
} else {
  df2 <- df2 %>%
    left_join(url_map %>% select(cop_num, URL) %>% distinct(),
              by = c(setNames("cop_num", COPNUM_COL)))
}

df2 <- df2 %>%
  mutate(
    URL = dplyr::coalesce(
      URL,
      source_pdf_http,
      vapply(source_pdf_file, github_url_from_file, character(1))
    )
  )

message("\nTop cop_num counts (first 25):")
print(df2 %>% count(.data[[COPNUM_COL]], sort = TRUE) %>% head(25))

message("\nDistinct URLs per cop_num (first 25):")
print(df2 %>% distinct(cop_num = .data[[COPNUM_COL]], URL) %>% arrange(as.numeric(cop_num)) %>% head(25))

message("\nNon-missing URL rows: ", sum(!is.na(df2$URL)))
message("Missing URL rows    : ", sum(is.na(df2$URL)))

# Check URL validity 
url_check <- df2 %>%
  distinct(URL) %>%
  mutate(
    is_http = is_http_url(URL),
    http_status = NA_integer_
  )

if ("httr" %in% rownames(installed.packages())) {
  safe_head <- function(u) {
    if (!is_http_url(u)) return(NA_integer_)
    out <- try(httr::HEAD(u, httr::timeout(15)), silent = TRUE)
    if (inherits(out, "try-error")) return(NA_integer_)
    httr::status_code(out)
  }
  url_check$http_status <- vapply(url_check$URL, safe_head, integer(1))
}

message("\nURL check summary (showing non-200 / NA http_status):")
print(url_check %>% filter(is_http & (is.na(http_status) | http_status >= 400)))

# Save (CSV + XLSX)
readr::write_csv(df2, OUTPUT_CSV, na = "")
writexl::write_xlsx(list(data = df2, url_check = url_check), OUTPUT_XLSX)

message("\nDONE")
message("Saved CSV : ", OUTPUT_CSV)
message("Saved XLSX: ", OUTPUT_XLSX)
