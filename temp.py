import pandas as pd



df_nodes = pd.read_csv("C:\\Users\\d.muehlfeld\\Berechnungsdaten\\11_Spechbach_RNAB_Node.csv")
df_pipes = pd.read_csv("C:\\Users\d.muehlfeld\\Berechnungsdaten\\11_Spechbach_RNAB_Pipes.csv")




"""
wenn ABGABENGE_E =!2
    egal
wenn ABGABENGE_E str=2
    suche die Leitung, wo der gleiche Knoten der Start list
        wenn beide Leitungen die gleiche Bauart haben
            verschelze die Leitungen
            sonst nichts machen
"""