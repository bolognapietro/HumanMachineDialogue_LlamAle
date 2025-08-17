import pandas as pd
import json
import re
from typing import Dict, Any, List, Optional
from copy import deepcopy
from rapidfuzz import process, fuzz


class BeerDataset:
    def __init__(self, path: str = "dataset/beer_data.csv"):
        """
        Loads and queries beer metadata from a CSV source.
        """
        self.data = pd.read_csv(path)
        self.path = path
        self._limit = len(self.data)

    def filter_by_intent(self, slots: Dict[str, Any], intent: str, top_k: int = 5) -> Optional[str]:
        """
        Filters beer dataset based on extracted slots.

        Args:
            slots (dict): Extracted slot values.
            intent (str): User intent (used to drive sort/filter logic).
            top_k (int): Max number of results to return.

        Returns:
            Optional[str]: JSON string with beer list, or None.
        """
        df = deepcopy(self.data)

        for key, value in slots.items():
            if value is None or value == "null":
                continue
            if key == "style":
                df = self._filter_by_style(value, df)
            elif key == "abv":
                df = self._filter_by_abv(value, df)
            elif key == "ibu":
                df = self._filter_by_ibu(value, df)
            elif key == "rating":
                df = self._filter_by_rating(value, df)
            elif key == "name":
                df = self._filter_by_name(value, df)
            elif key == "brewery":
                df = self._filter_by_brewery(value, df)

        if df.empty:
            return None

        if intent == "get_top_rated":
            df = df.sort_values(by="Rating", ascending=False)

        return self._format_json(df.head(top_k))

    def record_user_rating(self, slots: Dict[str, Any]) -> Optional[Dict]:
        """
        Updates rating/comment for a specific beer entry.

        Args:
            slots (dict): Contains keys 'name', 'rating', 'comment'

        Returns:
            dict or None: Confirmation payload, or None if failed.
        """
        name = slots.get("name")
        rating = slots.get("rating")
        comment = slots.get("comment")

        if name is None or rating is None:
            return None

        names = self.data["Name"].dropna().unique()
        matches = process.extract(name, names, scorer=fuzz.token_sort_ratio, score_cutoff=85, limit=5)

        if not matches:
            return None

        matched_name = matches[0][0]
        indices = self.data[self.data["Name"] == matched_name].index

        if indices.empty:
            return None

        if "User Rating" not in self.data.columns:
            self.data["User Rating"] = None
        if "User Comment" not in self.data.columns:
            self.data["User Comment"] = None

        self.data.loc[indices, "User Rating"] = rating
        if comment:
            self.data.loc[indices, "User Comment"] = f'"{comment}"'

        self.data.to_csv(self.path, index=False)

        row = self.data.loc[indices[0]]
        return {
            "beer": {
                "name": row["Name"],
                "brewery": row.get("Brewery", "Unknown"),
                "new_rating": rating,
                "comment": comment
            }
        }

    def _format_json(self, df: pd.DataFrame) -> str:
        beers = []
        df_dict = df.to_dict()

        for row_id in df_dict["Id"]:
            beers.append({
                "id": row_id,
                "name": df_dict["Name"].get(row_id),
                "full_name": df_dict["Beer Full Name"].get(row_id),
                "style": df_dict["Style"].get(row_id),
                "brewery": df_dict["Brewery"].get(row_id),
                "description": df_dict["Description"].get(row_id),
                "abv": df_dict["ABV"].get(row_id),
                "min_ibu": df_dict["Min IBU"].get(row_id),
                "max_ibu": df_dict["Max IBU"].get(row_id),
                "rating": df_dict["Rating"].get(row_id)
            })

        return json.dumps({"beers": beers}, indent=4)

    def _filter_by_abv(self, level: str, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["ABV"] = pd.to_numeric(df["ABV"], errors="coerce")
        df = df.dropna(subset=["ABV"])

        bounds = {
            "low": (0.0, 4.9),
            "medium": (5.0, 7.9),
            "high": (8.0, 100.0)
        }

        low, high = bounds.get(level, (0, 100))
        return df[(df["ABV"] >= low) & (df["ABV"] <= high)]

    def _filter_by_ibu(self, level: str, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["Min IBU"] = pd.to_numeric(df["Min IBU"], errors="coerce")
        df["Max IBU"] = pd.to_numeric(df["Max IBU"], errors="coerce")
        df = df.dropna(subset=["Min IBU", "Max IBU"])

        ranges = {
            "low": (0, 20),
            "medium": (21, 60),
            "high": (61, 120)
        }

        low, high = ranges.get(level, (0, 120))
        return df[(df["Max IBU"] >= low) & (df["Min IBU"] <= high)]

    def _filter_by_rating(self, val: Any, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.dropna(subset=["Rating"])

        if isinstance(val, (list, tuple)) and len(val) == 2:
            low, high = float(val[0]), float(val[1])
            return df[(df["Rating"] >= low) & (df["Rating"] <= high)]
        try:
            return df[df["Rating"] >= float(val)]
        except:
            return df

    def _filter_by_style(self, query: str, data: pd.DataFrame) -> pd.DataFrame:
        q = re.sub(r"\s+", " ", query.strip().lower())
        tokens = re.findall(r"\w+", q)

        col = data["Style"].dropna().str.lower()

        mask = col.notna()
        for token in tokens:
            mask &= col.str.contains(re.escape(token), na=False)

        substr_styles = data["Style"][mask].unique().tolist()
        df_substr = data[data["Style"].isin(substr_styles)] if substr_styles else pd.DataFrame()

        unique_styles = data["Style"].dropna().unique()
        fuzzy_styles = []
        for threshold in [90, 85, 80, 75]:
            matches = process.extract(q, unique_styles, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
            if matches:
                fuzzy_styles = [m[0] for m in matches]
                break

        df_fuzzy = data[data["Style"].isin(fuzzy_styles)] if fuzzy_styles else pd.DataFrame()

        if not df_substr.empty and not df_fuzzy.empty:
            intersect = pd.merge(df_substr, df_fuzzy, how="inner")
            return intersect if not intersect.empty else pd.concat([df_substr, df_fuzzy]).drop_duplicates()
        return df_substr or df_fuzzy or data.iloc[0:0]

    def _filter_by_name(self, query: str, data: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
        df = data.copy()
        name = query.strip().lower()

        df["Name_clean"] = df["Name"].astype(str).str.lower()
        df["Name_clean"] = df["Name_clean"].str.replace(r"\(.*?\)", "", regex=True)
        df["Name_clean"] = df["Name_clean"].str.replace(r"[^\w\s]", "", regex=True)
        df["Name_clean"] = df["Name_clean"].str.replace(r"\s+", " ", regex=True).str.strip()

        while True:
            matches = process.extract(name, df["Name_clean"], scorer=fuzz.ratio, limit=self._limit, score_cutoff=threshold)
            matched = [m[0] for m in matches]
            result = df[df["Name_clean"].isin(matched)]

            if result.empty and threshold > 30:
                threshold -= 5
            elif len(result) > 10 and threshold < 100:
                threshold += 1
            else:
                break

        return result

    def _filter_by_brewery(self, query: str, data: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
        df = data.copy()

        while True:
            matches = process.extract(query, df["Brewery"], scorer=fuzz.ratio, limit=self._limit, score_cutoff=threshold)
            matched = [m[0] for m in matches]
            result = df[df["Brewery"].isin(matched)]

            if result.empty and threshold > 30:
                threshold -= 5
            elif len(result) > 10 and threshold < 100:
                threshold += 1
            else:
                break

        return result