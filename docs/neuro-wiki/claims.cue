package neurowiki

// One line of claims.jsonl is checked against #Claim. CUE checks the shape and
// evidence-specific fields; check_neuro_wiki.py checks relationships across files.
#Text: string & =~"\\S"
#ClaimID: string & =~"^N-[0-9]{3,}$"
#Dataset: string & =~"^[a-z][a-z0-9-]*:v[0-9]+\\.[0-9]+$"
#URL: string & =~"^https?://[^ ]+$"
#RunJSON: string & =~"^runs/[^ ]+\\.json$"
#RunReport: string & =~"^runs/[^ ]+\\.md$"

#Base: {
    id:             #ClaimID
    subject:        #Text
    predicate:      string & =~"^[a-z][a-z0-9_]*$"
    object:         #Text
    evidence_type:  "published_experiment" | "dataset_annotation" | "connectome_edge" | "model_intervention" | "hypothesis"
    source:         #Text
    source_locator: #Text
    scope:          #Text
    limitation:     #Text
}

#Published: {
    #Base
    evidence_type: "published_experiment"
    source:        #URL
}

#Annotation: {
    #Base
    evidence_type: "dataset_annotation"
    dataset:       #Dataset
    source:        #URL | #RunJSON
    body_id:       int & >0
    side:          "L" | "R" | "midline" | "bilateral"
}

#Edge: {
    #Base
    evidence_type: "connectome_edge"
    dataset:       #Dataset
    source:        #URL | #RunJSON
    pre_body_id:   int & >0
    post_body_id:  int & >0
    synapse_count: int & >0
    filter:        #Text
    query:         #Text
}

#Model: {
    #Base
    evidence_type: "model_intervention"
    dataset:       #Dataset
    source:        #RunJSON
    report:        #RunReport
}

#Hypothesis: {
    #Base
    evidence_type: "hypothesis"
    source:        #URL | #RunJSON
    test_plan:     #Text
}

#Claim: #Published | #Annotation | #Edge | #Model | #Hypothesis
