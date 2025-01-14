def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_data_df",
                "description": "Get data from the databaase and return a pandas df",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string"},
                    },
                    "required": ["sql_query"],
                },
            },
        }
    ]
