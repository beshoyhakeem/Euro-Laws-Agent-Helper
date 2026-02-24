from src.graph.nodes import build_app_graph

app_graph = build_app_graph()

if __name__ == "__main__":

    q = """"
    Hello everyone, my divorce was finalized in 2024. And my ex is selling her property which we split during divorce. But we both are on deed till date when selling. There’s an offer in place. The property is being sold for profit close to 250K. And all proceeds will be going to my Ex. I have been told I have to be there for signing paperwork next week. 
1). How do I make sure the buyer doesn’t sue me in the future after buying the property for any kind of reasons. 
2). How do I make sure my ex is responsible for capital gains I’m not held responsible for those gains to pay tax.
Thankyou 🙏🏾
    
    """
    result = app_graph.invoke({"question": q})
    final_answer = result["answer"]

    print(final_answer)