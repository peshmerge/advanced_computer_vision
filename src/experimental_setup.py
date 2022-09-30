from  src.model import cnnLSTM
# from dataset import Ego4D
from torch import nn
from torch import optim
import torch
from datetime import datetime

"""
What is in here?
The function to run the training loop, and validate.

"""




def train(train_loader, model_settings,validation_loader):
    optimizer = model_settings["optimizer"]
    loss = model_settings["loss"]
    epochs = model_settings["epoch"]
    model = model_settings["model"]
    device = model_settings["device"]
    runningloss = 0.0

    correctStats = {}
    print (f"[+]Training on {device}")
    correct = 0
    validation_results = None #This will be the best accuracy of the model

    total,correct = 0,0

    for epoch in range(1,epochs):
        print(f"Starting epoch : {epoch}/{epochs}")
        model.train()
        for iteration ,(input,target) in enumerate(train_loader,1):
 
            input = input.type(torch.FloatTensor).to(device)
            # print (torch.typename(input))
            target = torch.Tensor([target]).type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            output = model(input)
            # print (output)
            # print(output.size())
            # acc = 100 * (output.detach().argmax(1) == ).cpu().numpy().mean())
            # targetLabel = torch.fill(())


            currLoss = loss(output,target)
            # print(f"Sequence accuracy : {()}")
            currLoss.backward()
            runningloss+=currLoss.item() * input.size(0)

            _,predicted = torch.max(output.data,1)
            correct+=(target.cpu()==predicted.cpu()).sum()
            total+=target.size(0)
            optimizer.step()

            

            # predlabel = torch.argmax(output.detach()).item()
        
            # same = (predlabel==target).item()
            # print (same)
            

            # print (f"Loss : {runningloss/len(train_loader)}")
            if iteration%10==0:
                print (f"Running loss : {runningloss/iteration}")
        

        training_accuracy = (correct/len(train_loader))*100
        print (f"[Training]accuracy : {training_accuracy}%")
        
        training_loss = (runningloss/len(train_loader))
        print(f"[Training]loss : {training_loss}")
        runningloss = 0
        # print("[Training]Correct class predictions")
        # print(correctStats)
        # correctStats = {}
        correct= 0
        torch.cuda.empty_cache() 
        print ("[+]Starting validation")
        validation_results = validation(validation_loader,model,device,
                                        validation_results,training_accuracy,training_loss)


    

def validation(validation_loader,model,device,validation_results,training_accuracy,training_loss):
    correct = 0
    totalSamples = len(validation_loader)
    correctStats = {}
    model.eval()
    with torch.no_grad():
        for input,target in validation_loader:
            input = input.type(torch.FloatTensor).to(device)
            # print (input)
            # print (input.size())
            target = torch.Tensor([target]).type(torch.LongTensor).to(device)
            # print (target)
            output = torch.argmax(model(input)).item()
            if output==target:
                correct+=1
                if output not in correctStats.keys():
                    correctStats[output] = 1
       
                else:
                    correctStats[output] = correctStats[output]+1
               
             
    print("[Validation]Correct predicted classes")
    print (correctStats)
    accuracy = (correct/totalSamples)*100
    print (f"[Validation]Total correct : {correct}/{totalSamples}")
    print (f"[Validation]Accuracy : {accuracy}%")
    time_stamp = datetime.timestamp(datetime.now())
    torch.save(model,f"saved-model/{time_stamp}-val_acc-{accuracy:.3f}-acc-{training_accuracy:.3f}-loss-{training_loss:.3f}.bin")
    
    if validation_results:
        if accuracy>validation_results:
            time_stamp = datetime.timestamp(datetime.now())
            print(f"[+]Saving model at validation accuracy of : {accuracy}%")
            # torch.save(model,f"saved-model/{time_stamp}-val_acc-{accuracy:.3f}-acc-{training_accuracy:.3f}-loss-{training_loss:.3f}.bin")
            print("[+]Model successfully saved!")
            torch.cuda.empty_cache()
            return accuracy #This is going to be the new validation accuracy
        else:
            print("[+]No improvement in model validation")
            return validation_results
    else:
        torch.cuda.empty_cache()
        return accuracy    




