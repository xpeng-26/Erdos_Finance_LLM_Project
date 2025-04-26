import time
from llama_cpp import Llama
import os
import logging
from pathlib import Path
import pandas as pd
import json
import re
from jsonschema import validate, ValidationError
from utils.database.schema import create_schema
from utils.database.manager import DatabaseManager


class LlmAnalyst:
    """
    Class for inferencing AI sentiment score and trading advisory using the provided news data
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # will load the LLM model
        self.model_path = config["feature"]["llm_model"]["model_path"]
        self.lib_path = config["feature"]["llm_model"]["lib_path"]
        self.max_tokens = 2**14

        # will load news data from the database
        self.news_db_path = (
            Path(config["info"]["local_data_path"])
            / "data_raw"
            / config["info"]["db_news_name"]
        )
        self.news_db_manager = DatabaseManager(self.news_db_path)

        # will write the inferenced data to the daily stock database
        self.daily_db_path = (
            Path(config["info"]["local_data_path"])
            / "data_raw"
            / config["info"]["db_name"]
        )
        self.daily_db_manager = DatabaseManager(self.daily_db_path)

        # define the schema for the JSON output
        self.json_schema = {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "news_summary": {"type": "string"},
                "sentiment_score": {
                    "type": "object",
                    "properties": {
                        "short_term": {"type": "number"},
                        "mid_term": {"type": "number"},
                        "long_term": {"type": "number"},
                    },
                    "required": ["short_term", "mid_term", "long_term"],
                },
                "adjustment": {
                    "type": "object",
                    "properties": {
                        "mid_term": {"type": "number"},
                        "long_term": {"type": "number"},
                    },
                    "required": ["mid_term", "long_term"],
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "business_structure_analysis": {"type": "string"},
                        "market_reaction": {"type": "string"},
                        "overall_analysis": {"type": "string"},
                    },
                    "required": [
                        "business_structure_analysis",
                        "market_reaction",
                        "overall_analysis",
                    ],
                },
            },
            "required": [
                "symbol",
                "news_summary",
                "sentiment_score",
                "adjustment",
                "analysis",
            ],
        }

    def flatten_json(self, instance):
        """Flatten the JSON data into a dataframe"""
        flattened_data = {
            "symbol": instance["symbol"],
            "news_summary": instance["news_summary"],
            "sentiment_short": instance["sentiment_score"]["short_term"],
            "sentiment_mid": instance["sentiment_score"]["mid_term"],
            "sentiment_long": instance["sentiment_score"]["long_term"],
            "adjustment_mid": instance["adjustment"]["mid_term"],
            "adjustment_long": instance["adjustment"]["long_term"],
            "overall_analysis": instance["analysis"]["overall_analysis"],
            "business_structure_analysis": instance["analysis"][
                "business_structure_analysis"
            ],
            "market_reaction": instance["analysis"]["market_reaction"],
        }
        return pd.DataFrame([flattened_data])

    def _load_news_data(self, symbol):
        """Load news data from the database for one symbol"""
        try:
            news_df = self.news_db_manager.query(
                f"SELECT * FROM news WHERE symbol = '{symbol}' AND length_summary > 50 AND relevance_score > 0.05 ORDER BY datetime"
            )
            # Combine the title and summary columns to create the news column
            news_df["news"] = (
                "Title: "
                + news_df["title"]
                + ". Summary: "
                + news_df["summary"]
                + ". Source: "
                + news_df["source"]
            )
            # Return news_df with three columns: datetime, date, news
            news_df["date"] = pd.to_datetime(news_df["datetime"]).dt.strftime(
                "%Y-%m-%d"
            )
            news_df = news_df[["datetime", "date", "news"]]
            unique_dates = news_df["date"].unique().tolist()
            return news_df, unique_dates
        except Exception as e:
            self.logger.error(f"Error loading news data: {e}")
            raise e

    def _load_daily_data(self):
        """Load daily stock data from the database"""
        try:
            daily_df = self.daily_db_manager.query(
                "SELECT * FROM daily_prices ORDER BY symbol, date"
            )
            self.symbols = daily_df["symbol"].unique()
            return daily_df
        except Exception as e:
            self.logger.error(f"Error loading daily price data: {e}")
            raise e

    def _load_llm_model(self):
        """Load the LLM model"""
        try:
            self.model = Llama(
                model_path=os.path.expanduser(self.model_path),
                lib_path=os.path.expanduser(self.lib_path),
                n_ctx=2**14,
                n_threads=6,
                n_gpu_layers=32,
                use_mlock=True,
                use_mmap=True,
                seed=42,
                verbose=False,
            )
        except Exception as e:
            self.logger.error(f"Error loading LLM model: {e}")
            raise e

    def _truncate_text_to_tokens(self, text):
        """Truncate the text to the maximum number of tokens"""
        tokens = self.model.tokenize(text.encode("utf-8"), add_bos=False)
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
            self.logger.warning(
                f"News length exceeds the context window of the LLM. Truncated to {self.max_tokens} tokens."
            )
        return self.model.detokenize(tokens).decode("utf-8", errors="ignore")

    def _news_summary(self, news, symbol, date):
        news = "  ".join(news)
        news = self._truncate_text_to_tokens(text=news)
        """Summarize the news data if multiple news items are available for a single day"""
        # analyze the news data and inference the sentiment score and trading advisory
        prompt = f"""
        {news}

        Summarize the above news into one concise paragraph. Only include the most important points, and do not repeat any details from the original news. 
        Do not include titles, summaries, or source references. Ensure the summary is plain text with no markdown, code blocks, or unnecessary explanations. 
        Ensure that the output is in ONE single paragraph with no extra line breaks or spaces.
        """
        summary = self.model(
            prompt,
            max_tokens=500,
            temperature=0.0,
            top_p=0.95,
            top_k=30,
            repeat_penalty=1.2,
        )
        return summary["choices"][0]["text"]

    def _extract_json_from_text(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        return None

    def _inference_validate_json(self, prompt, symbol, date, max_retries=5, delay=0.1):
        """Validate the JSON string"""
        for attempt in range(1, max_retries + 1):
            output = self.model(
                prompt,
                max_tokens=1000,
                temperature=0.1,
                top_p=0.95,
                top_k=30,
                repeat_penalty=1.1,
            )["choices"][0]["text"]
            try:
                output = self._extract_json_from_text(output)
                # Decode the JSON string
                json_data = json.loads(output)
                # Validate the JSON data
                validate(instance=json_data, schema=self.json_schema)
                return json_data
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(
                    f"For Symbol {symbol} on {date}, invalid JSON response. Retrying..."
                )
                if attempt < max_retries:
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Max retries exceeded. Skip the inference for symbol {symbol} on {date}."
                    )
                    return None

    def _inference(self, news_summary, symbol, date):
        """Inference the sentiment score and trading advisory"""

        # prompt
        news = f"Original news of Company {symbol}: {news_summary}"

        prompt = f"""
                (role) You are a senior U.S. equity analyst specializing in {symbol}.

                For this news:{news}, Your task is to analyze the news in a structured way by following the steps below:

                == Analysis Phase ==

                (analysis 1: business_structure_analysis)  
                Based on your understanding of the company's business structure, identify whether the affected product/business belongs to a core or peripheral segment, and assess the estimated impact on overall profitability (high/medium/low) and duration (short/medium/long).

                (analysis 2: market_reaction)  
                Consider the current expectations of investors regarding this company's stock price. Analyze how this news might shift the sentiment of bullish and bearish investors and what behavioral change it may cause in the market.

                == Output Phase ==

                Based on your analysis above, complete the following tasks:

                (action 1: overall_analysis)  
                Summarize your overall assessment of the news's impact on the company (max 200 words) to represent your perspective.

                (action 2: sentiment_score)  
                Provide a sentiment score for investors with different holding periods. Score range: [-1.0, 1.0], using the following standards:

                - |score| > 0.7 means clear and lasting impact  
                - 0.3 < |score| ≤ 0.7 means moderate impact  
                - |score| ≤ 0.3 means mild or uncertain impact

                Output scores for:  
                - short_term (1–30 days)  
                - mid_term (30–180 days)  
                - long_term (180+ days)

                (action 3: adjustment)  
                Assume the client already holds a small position (less than 20% of the total portfolio) in this stock and is a medium/low frequency trader. Provide adjustment suggestions for:
                - mid_term (30+ days)  
                - long_term (180+ days)

                Express your suggestion as a percentage of the total portfolio value:
                - Use a **positive** value to recommend increasing position (e.g., 0.05 = +5%) 
                - Use a **negative** value to recommend reducing position (e.g., -0.03 = -3%)  
                - Use 0 for no recommend adjustment, if the news is not relevant or the impact is unclear 

                == Output Format ==

                You must return a raw JSON object in the exact format below, with no extra explanation or natural
                language text. Do not wrap it in markdown, explanations, ``` or any formatting.

                {{
                    "symbol": "{symbol}",
                    "news_summary": "{news_summary}",
                    "sentiment_score": {{
                        "short_term": <float>,
                        "mid_term": <float>,
                        "long_term": <float>
                    }},
                    "adjustment": {{
                        "mid_term": <float>,
                        "long_term": <float>
                    }},
                    "analysis": {{
                        "business_structure_analysis": "<max 100 words>",
                        "market_reaction": "<max 100 words>",
                        "overall_analysis": "<max 200 words>"
                    }}
                }}
                """
        inference = self._inference_validate_json(prompt, symbol, date)

        return inference

    def run(self):
        """Run the LLM analyst"""

        # if the table does not exist, create a new one
        if not self.daily_db_manager.check_table_exists("news_factors"):
            self.daily_db_manager.setup_table(create_schema("news_factors"))
            self.logger.info("News factors table created")
        else:
            self.logger.info("News factors table already exists")
            self.daily_db_manager.delete_table("news_factors")
            self.daily_db_manager.setup_table(create_schema("news_factors"))
            self.logger.info("News factors table recreated")

            # load the LLM model
        self._load_llm_model()

        # load the daily stock data
        price_df = self._load_daily_data()

        # loop over symbols and datetimes to inferenece the sentiment score and trading advisory
        self.logger.info(
            "Start inferencing the sentiment score and trading advisory for symbols: {}...".format(
                self.symbols
            )
        )
        for symbol in self.symbols:
            symbol_news_df, symbol_dates = self._load_news_data(symbol)
            empty_df = pd.DataFrame(
                {
                    "symbol": [symbol],
                    "news_summary": [None],
                    "sentiment_short": [None],
                    "sentiment_mid": [None],
                    "sentiment_long": [None],
                    "adjustment_mid": [None],
                    "adjustment_long": [None],
                    "overall_analysis": [None],
                    "business_structure_analysis": [None],
                    "market_reaction": [None],
                }
            )
            self.logger.info(
                "Inferencing the sentiment score and trading advisory for symbol: {}, from dates: {} to {}...".format(
                    symbol, min(symbol_dates), max(symbol_dates)
                )
            )

            date_count = 0
            for date in symbol_dates:
                raw_news = symbol_news_df[symbol_news_df["date"] == date][
                    "news"
                ].unique().tolist()
                # insert a null row to the database
                if len(raw_news) == 0:
                    llm_inference = empty_df
                # use the LLM to inference the sentiment score and trading advisory
                else:
                    llm_summary = self._news_summary(raw_news, symbol, date)
                    print(llm_summary)
                    llm_inference = self._inference(llm_summary, symbol, date)
                    print(llm_inference)
                    if llm_inference is not None:
                        llm_inference = self.flatten_json(llm_inference)
                    else:
                        llm_inference = empty_df

                llm_inference["date"] = date

                # write back to the database for news_factors table
                self.daily_db_manager.upsert(
                    data=llm_inference,
                    table_name="news_factors",
                    primary_keys=["symbol", "date"],
                )
                # print the progress
                date_count += 1
                self.logger.info(
                    "Inferencing progress: {} is done, {}/{}".format(
                        date, date_count, len(symbol_dates)
                    )
                )
