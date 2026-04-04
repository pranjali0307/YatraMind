import pandas as pd

def load_places():
    df = pd.read_csv("data/Top Indian Places to Visit.csv")

    df = df.fillna("")
    df.columns = df.columns.str.strip()

    # Convert everything to string (VERY IMPORTANT FIX)
    for col in df.columns:
        df[col] = df[col].astype(str)

    # Detect category column
    category_col = None
    for col in df.columns:
        if col.lower() in ["category", "type", "tags", "theme"]:
            category_col = col
            break

    if category_col is None:
        category_col = df.columns[2]

    # Detect best time column
    best_time_col = None
    for col in df.columns:
        if "time" in col.lower():
            best_time_col = col
            break

    # Create combined column
    if best_time_col:
        df["combined"] = (
            df["Name"] + " " +
            df["State"] + " " +
            df[category_col] + " " +
            df[best_time_col]
        )
    else:
        df["combined"] = (
            df["Name"] + " " +
            df["State"] + " " +
            df[category_col]
        )

    df["Category"] = df[category_col]

    places = df.to_dict("records")

    return places, df