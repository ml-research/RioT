from .rrr_loss import RRRLoss, HorizonRRRIGLoss, HorizonRRRFIGLoss,HorizonRRRFTIGLoss,  RRRFIGLoss, RRRIGLoss, RRRFTIGLoss


Forecasting_Right_Reason_Loss =  HorizonRRRIGLoss | HorizonRRRFIGLoss | HorizonRRRFTIGLoss
Classification_Right_Reason_Loss = RRRFIGLoss | RRRFTIGLoss

Right_Reason_Loss = RRRLoss | RRRIGLoss | Forecasting_Right_Reason_Loss | Classification_Right_Reason_Loss
