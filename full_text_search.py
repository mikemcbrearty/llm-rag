import json

import requests


class FullTextSearch:

    solr_url = "http://localhost:8983"
    collection_name = "aci_318_08_sections"

    def __init__(self, verbose: bool=False):
        self.verbose = verbose

        self.__deleteCollection()
        self.__createCollection()
        self.__defineSchema()
        self.__indexACISections()
        self.__commitChanges()


    def __deleteCollection(self) -> None:
        url = f"{self.solr_url}/solr/admin/collections"
        params = {
            'action': 'DELETE',
            'name': self.collection_name,
        }
        response = requests.get(url, params=params)
        if self.verbose:
            print(f"deleteCollection: {response.status_code}")
            print(response.text)


    def __createCollection(self) -> None:
        url = f"{self.solr_url}/api/collections"
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'name': self.collection_name,
            'numShards': 1,
            'replicationFactor': 1,
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if self.verbose:
            print(f"createCollection: {response.status_code}")
            print(response.text)


    def __defineSchema(self) -> None:
        url = f"{self.solr_url}/api/collections/{self.collection_name}/schema"
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'add-field': [
                {'name': "text", 'type': 'text_general'},
                {'name': "section", 'type': 'string'},
            ]
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if self.verbose:
            print(f"defineSchema: {response.status_code}")
            print(response.text)


    def __indexACISections(self) -> None:
        aci_doc_file = "documents/318_08_sections.txt"
        with open(aci_doc_file, 'r') as f:
            for line in f:
                i = line.find(" ")
                section = line[:i]
                self.__indexDocument(text=line, section=section)


    def __indexDocument(
        self,
        text: str="text",
        section: str="999.999",
        verbose: bool=False,
    ) -> None:
        url = f"{self.solr_url}/api/collections/{self.collection_name}/update"
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'text': text,
            'section': section,
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if verbose or self.verbose:
            print(f"indexDocument: {response.status_code}")
            print(response.text)


    def __commitChanges(self) -> None:
        url = f"{self.solr_url}/solr/{self.collection_name}/update"
        params = {
            'commit': 'true',
        }
        response = requests.get(url, params=params)
        if self.verbose:
            print(f"commitChanges: {response.status_code}")
            print(response.text)


    def query(self, query_text: str, top_k: int=3, verbose: bool=False) -> list[str]:
        url = f"{self.solr_url}/solr/{self.collection_name}/select"

        formatted_query_text = " OR ".join([f"text:{term}" for term in query_text.split(' ')])
        params = {
            'q': formatted_query_text,
            'rows': str(top_k),
        }
        response = requests.get(url, params=params)
        response_text = [elem['text'][0] for elem in json.loads(response.text)['response']['docs']]
        if verbose or self.verbose:
            print(f"query: {response.status_code}")
            print(response.text)
        return response_text
