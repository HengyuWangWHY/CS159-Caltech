import argparse
import pickle

from openai import OpenAI
import numpy as np
from pandas import DataFrame
from tdc.benchmark_group import dti_dg_group


def create_openai_client():
    '''
    Returns an instance of an OpenAI API client
    using the API key in the text file ./openai_api_key
    '''
    with open('./openai_api_key') as f:
        openai_api_key = f.readline().strip()
    return OpenAI(api_key = openai_api_key)


def load_dti_data() -> tuple[DataFrame, DataFrame]:
    '''
    Returns a tuple of two pandas dataframes
    containing the train/val data and the test data.
    If the data is not already at ./data,
    it will be downloaded and stored there.
    '''
    group = dti_dg_group()
    benchmark = group.get('BindingDB_Patent')
    return (benchmark['train_val'], benchmark['test'])


def extract_trainval_percentiles(train_val: DataFrame) -> str:
    '''
    Returns a string containing the percentiles of the training data
    
    Input
        train_val: the pandas dataframe for the train/val set
    
    Returns
        percentiles: a string version of a list of tuples
                     containing the percentile rank and value
                     for all 100 Kd percentiles in the train/val set
    '''
    percentiles = [
        (
            p,
            round(
                np.percentile(
                    train_val['Y'].to_list(),
                    [i for i in range(101)]
                )[p],
                6
            )
        )
        for p in range(101)
    ]
    return percentiles


def format_input(
    data: DataFrame, num_samples: int, seed: int = 42
) -> tuple[str, list]:
    '''
    Returns a tuple of containing a string with lines of SMILES strings
    and amino acid sequence pairs,
    and a list of associated Kd values for each row
    
    Input
        data: a pandas dataframe containing the data to predict on
        num_samples: the number of samples to take from the data
        seed: the seed for random sampling from the data
        
    Returns
        x: a string with num_samples rows of SMILES-AASeq pairs
        y: a list of corresponding Kd values
    '''
    rng = np.random.default_rng(seed=seed)
    num_samples = num_samples
    mol = data['Drug'].to_list()
    prot = data['Target'].to_list()
    kd = data['Y'].to_list()
    x = ''
    y = []
    picks = rng.choice(np.arange(len(data)), replace=False, size=num_samples)
    for i in range(num_samples):
        x = x + mol[picks[i]] + ' ' + prot[picks[i]]
        if num_samples > 1:
            x = x + ' \n'
        y.append(kd[picks[i]])
    return (x, y)


def extract_predictions(res: str) -> list[float]:
    '''
    Returns list of Kd predictions extracted from the LLM response.
    Predictions are expected to be of the form >>> pred <<<,
    where pred is castable to float. Throws an assertion error
    if no predictions are found in the correct format.
    
    Input
        res: Message output from LLM. Expected to have predictions
             of the form >>> pred <<<, where pred is castable to float
             
    Returns
        pred: list of Kd predictions extracted from response
    '''
    pred = []
    start = 0
    for i in range(res.count('>>>')):
        begin = res.find('>>>', start) + 4
        end = res.find('<<<', begin) - 1
        pred.append(float(res[begin : end]))
        start = end + 4
    assert len(pred) > 0, (
        "Response did not have any predictions of the form >>> pred <<<."
    )
    return pred


def score_predictions(pred: list[float], true: list[float]) -> float:
    '''
    Returns the Pearson Correlation Coefficient score
    used by TDC for the DTI task.
    Can accept a full list of true values and a partial list of predictions
    in case the model does not make a prediction for every pair.
    In this case, it computes the score
    only on pairs for which a prediction exists.
    
    Input
        pred: list of predicted Kd values
        true: list of actual Kd values
    
    Returns
        score: Pearson Correlation coefficient of the inputs
    '''
    assert len(pred) <= len(true), (
        f'There are {len(pred)} predictions, '
        f'but only {len(true)} ground truth values.'
    )
    assert len(pred) >= len(true), (
        f'There are {len(true)} ground truth values, '
        f'but only {len(pred)} predictions.'
    )
    return np.corrcoef(pred, true[:len(pred)])[0,1]
    
    
def run_benchmark(client, x, y):
    notools_assistant = client.beta.assistants.create(
        model='gpt-4o',
        name='No-Tools',
    #     instructions=(
# '''You are an artificial super-intelligence capable of solving computational biology problems with very little information available to you.
# You are tasked with computing the dissociation constant (Kd) between a drug molecule given by its SMILES string and a target protein given by its amino acid sequence.
# Every query will consist of a number of rows, each of which has the SMILES string of a drug molecule, followed by a space, and then the amino acid sequence of the target protein, concluded by a newline.
# For each row, you must predict the dissociation constant Kd.
# The answers should be in units of nanomolar (nM).
# Please preface each answer with three less than signs and a space, and finish each answer with a space and three greater than signs, with only a numerical value inside.
# Here is an example of an input row:
# O=C(O)CCC(=O)C(=O)O MANDSGGPGGPSPSERDRQYCELCGKMENLLRCSRCRSSFYCCKEHQRQDWKKHKLVCQGSEGALGHGVGPHQHSGPAPPAAVPPPRAGAREPRKAAARRDNASGDAAKGKVKAKPPADPAAAASPCRAAAGGQGSAVAAEAEPGKEEPPARSSLFQEKANLYPPSNTPGDALSPGGGLRPNGQTKPLPALKLALEYIVPCMNKHGICVVDDFLGKETGQQIGDEVRALHDTGKFTDGQLVSQKSDSSKDIRGDKITWIEGKEPGCETIGLLMSSMDDLIRHCNGKLGSYKINGRTKAMVACYPGNGTGYVRHVDNPNGDGRCVTCIYYLNKDWDAKVSGGILRIFPEGKAQFADIEPKFDRLLFFWSDRRNPHEVQPAYATRYAITVWYFDADERARAKVKYLTGEKGVRVELNKPSDSVGKDVF
# If the query were to contain that input row, you would report the following answer for that row: >>> 900.0 <<<
# You must make your best estimate of a numerical value for each row, and make sure that each row has exactly one answer.
# The number of predictions in the specified format should be equal to the number of rows in the query.'''
#     ),
#     instructions=(
# '''You are an artificial super-intelligence capable of solving computational biology problems with very little information available to you.
# You are tasked with computing the dissociation constant (Kd) between a drug molecule given by its SMILES string and a target protein given by its amino acid sequence.
# A query will consist of a SMILES string of a drug molecule, followed by a space, and then the amino acid sequence of the target protein.
# You must predict the dissociation constant Kd in units of nanomolar (nM).
# You may output more than just the answer, but please preface each answer with three less than signs and a space, and finish each answer with a space and three greater than signs, with only a numerical value inside.
# Here is an example of an input:
# O=C(O)CCC(=O)C(=O)O MANDSGGPGGPSPSERDRQYCELCGKMENLLRCSRCRSSFYCCKEHQRQDWKKHKLVCQGSEGALGHGVGPHQHSGPAPPAAVPPPRAGAREPRKAAARRDNASGDAAKGKVKAKPPADPAAAASPCRAAAGGQGSAVAAEAEPGKEEPPARSSLFQEKANLYPPSNTPGDALSPGGGLRPNGQTKPLPALKLALEYIVPCMNKHGICVVDDFLGKETGQQIGDEVRALHDTGKFTDGQLVSQKSDSSKDIRGDKITWIEGKEPGCETIGLLMSSMDDLIRHCNGKLGSYKINGRTKAMVACYPGNGTGYVRHVDNPNGDGRCVTCIYYLNKDWDAKVSGGILRIFPEGKAQFADIEPKFDRLLFFWSDRRNPHEVQPAYATRYAITVWYFDADERARAKVKYLTGEKGVRVELNKPSDSVGKDVF
# If the query were to be that input, you would report the following answer: >>> 900.0 <<<
# You must make your best estimate at a numerical value for Kd.'''
#         ),
    instructions=(
'''You are an artificial super-intelligence capable of solving computational biology problems with very little information available to you.
You are tasked with computing the natural logarithm of the dissociation constant (Kd) between a drug molecule given by its SMILES string and a target protein given by its amino acid sequence.
A query will consist of a SMILES string of a drug molecule, followed by a space, and then the amino acid sequence of the target protein.
You must predict the log of the dissociation constant Kd, which is in units of nanomolar (nM).
You may output more than just the answer, but please preface each answer with three less than signs and a space, and finish each answer with a space and three greater than signs, with only a numerical value inside.
Here is an example of an input:
O=C(O)CCC(=O)C(=O)O MANDSGGPGGPSPSERDRQYCELCGKMENLLRCSRCRSSFYCCKEHQRQDWKKHKLVCQGSEGALGHGVGPHQHSGPAPPAAVPPPRAGAREPRKAAARRDNASGDAAKGKVKAKPPADPAAAASPCRAAAGGQGSAVAAEAEPGKEEPPARSSLFQEKANLYPPSNTPGDALSPGGGLRPNGQTKPLPALKLALEYIVPCMNKHGICVVDDFLGKETGQQIGDEVRALHDTGKFTDGQLVSQKSDSSKDIRGDKITWIEGKEPGCETIGLLMSSMDDLIRHCNGKLGSYKINGRTKAMVACYPGNGTGYVRHVDNPNGDGRCVTCIYYLNKDWDAKVSGGILRIFPEGKAQFADIEPKFDRLLFFWSDRRNPHEVQPAYATRYAITVWYFDADERARAKVKYLTGEKGVRVELNKPSDSVGKDVF
If the query were to be that input, you would report the following answer: >>> 6.802395 <<<
You must make your best estimate at a numerical value for ln(Kd).'''
        ),
        tools=[],
        temperature=0.2
    )

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role='user',
        content=x
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=notools_assistant.id
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    res = messages.dict()['data'][0]['content'][0]['text']['value']

    pred = extract_predictions(res)
    return pred

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nsamples', action='store', dest='num_samples', default=10, type=int
    )
    args = parser.parse_args()
    train_val, test = load_dti_data()
    client = create_openai_client()
    preds = []
    true = []
    for i in range(args.num_samples):
        x, y = format_input(data=test, num_samples=1, seed=i)
        preds.extend(run_benchmark(client, x, y))
        true.extend(y)
        if i % 100 == 99:
            print(i+1)
            pickle.dump(preds, open(f'./output/preds_{i+1}_log', 'wb'))
            pickle.dump(true, open(f'./output/true_{i+1}_log', 'wb'))            

    score = score_predictions(preds, true)
    log_score = score_predictions(np.log(preds), true)
    exp_score = score_predictions(preds, np.exp(true))
    print('Pearson Correlation Coefficient: {0:.3}'.format(score))
    print('With log: {0:.3}\n'.format(log_score))
    # print(np.vstack([np.log(preds), true]).T)