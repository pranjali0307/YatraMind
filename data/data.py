import pandas as pd


def load_places():

    # Load CSV
    df = pd.read_csv("data/Top Indian Places to Visit.csv")

    # Clean data
    df = df.fillna("")
    df.columns = df.columns.str.strip()

    # Convert all to string (important for AI search)
    for col in df.columns:
        df[col] = df[col].astype(str)

    # ----------------------------
    # Detect Name Column
    # ----------------------------

    name_col = None

    for col in df.columns:
        if col.lower() in ["name", "place", "destination"]:
            name_col = col
            break

    if name_col is None:
        name_col = df.columns[0]

    df["Name"] = df[name_col]


    # ----------------------------
    # Detect State Column
    # ----------------------------

    state_col = None

    for col in df.columns:
        if col.lower() in ["state", "location", "region"]:
            state_col = col
            break

    if state_col is None:
        state_col = df.columns[1]

    df["State"] = df[state_col]


    # ----------------------------
    # Detect Category Column
    # ----------------------------

    category_col = None

    for col in df.columns:
        if col.lower() in ["category", "type", "tags", "theme"]:
            category_col = col
            break

    if category_col is None:
        category_col = df.columns[2]

    df["Category"] = df[category_col]


    # ----------------------------
    # Detect Rating Column
    # ----------------------------

    rating_col = None

    for col in df.columns:
        if "rating" in col.lower():
            rating_col = col
            break

    if rating_col:
        df["Rating"] = df[rating_col]
    else:
        df["Rating"] = "4"


    # ----------------------------
    # Detect City Column (Optional)
    # ----------------------------

    city_col = None

    for col in df.columns:
        if col.lower() in ["city", "district", "location"]:
            city_col = col
            break

    if city_col:
        df["City"] = df[city_col]
    else:
        df["City"] = df["State"]


    # ----------------------------
    # Detect Best Time Column
    # ----------------------------

    best_time_col = None

    for col in df.columns:
        if "time" in col.lower() or "season" in col.lower():
            best_time_col = col
            break


    # ----------------------------
    # Create AI Search Column
    # ----------------------------

    if best_time_col:

        df["combined"] = (
            df["Name"] + " " +
            df["State"] + " " +
            df["City"] + " " +
            df["Category"] + " " +
            df[best_time_col]
        )

    else:

        df["combined"] = (
            df["Name"] + " " +
            df["State"] + " " +
            df["City"] + " " +
            df["Category"]
        )


    # Convert to dict
    places = df.to_dict("records")

    return places, df