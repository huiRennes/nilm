//  Main NILM Arduino sketch.
//
//  Continually sends the following over the serial port:
//    RMS line voltage (one phase only)
//    RMS current (both phases)
//    Calculated real power (both phases)
//    Calculated apparent power (both phases).
//
//  Automatic gain control of the analog front end is done every sample period.
//
//  Copyright (c) 2022 Lindo St. Angel

#include <Arduino.h>
#include "emonLibCM.h"

// Define MCU GPIOs for gain control bits.
#define CT1_G0 (byte) 42 // ATMEGA2560 pin 42 (PL7) - GPIO42 - CT1 GAIN0
#define CT1_G1 (byte) 43 // ATMEGA2560 pin 41 (PL6) - GPIO43 - CT1 GAIN1
#define CT2_G0 (byte) 49 // ATMEGA2560 pin 35 (PL0) - GPIO49 - CT2 GAIN0
#define CT2_G1 (byte) 48 // ATMEGA2560 pin 36 (PL1) - GPIO48 - CT2 GAIN1

// Define ADC channels.
#define NUM_ADC_I_CHAN 2 // Number of ADC channels used for current sensing
#define V_CHAN  (byte) 0 // Voltage sense
#define I1_CHAN (byte) 1 // Current sense 1
#define I2_CHAN (byte) 2 // Current sense 2

// Set ADC full scale range to 2.56 Volts.
#define ADC_FS (double) 2.56

// Analog front end gains.
#define LOW_GAIN  (double) -1.3
#define MID_GAIN  (double) -10.2
#define HIGH_GAIN (double) -63.0

// Define current transformer max reading. 
#define CT_MAX (double) 100 // 100 Amps rms.

// Define calibration constants for Emon lib power calculations. 
#define V_AMP_CAL       (double) 180.0
#define I1_AMP_CAL_LOW  (double) 110.9
#define I1_AMP_CAL_MID  (double) 14.0
#define I1_AMP_CAL_HIGH (double) 2.29
#define I1_PH_CAL       (double) 4.6
#define I2_AMP_CAL_LOW  (double) 110.9 // was 144.0
#define I2_AMP_CAL_MID  (double) 14.0
#define I2_AMP_CAL_HIGH (double) 2.29
#define I2_PH_CAL       (double) 4.2

// Constants for determining thresholds to switch gains. 
#define ADC_FS_VRMS (ADC_FS / (2.0 * sqrt(2.0))) // 0.905 Vrms
#define UPPER_THRESHOLD (double) 0.95
#define LOWER_THRESHOLD (double) 0.85

// Define current transformer gain control bit tuple
typedef struct gainBits
{
  int g1;
  int g0; // LSB
} gainBitsType;

// Create a gain control tuple for each current transformer
gainBitsType const ct1GainCtrl = {.g1 = CT1_G1, .g0 = CT1_G0};
gainBitsType const ct2GainCtrl = {.g1 = CT2_G1, .g0 = CT2_G0};

// Analog front end gain settings.
typedef enum gainSettings
{
  lowGain,
  midGain,
  highGain,
  invalidGain
} gainSettingType;

// AGC inputs.
typedef struct agc
{
  byte channel;
  double vRMS;
  double iRMS;
  double apparentPower;
} agcType;

// Function prototypes.
gainSettingType GetGainSettings(gainBitsType gainCtrl);
double GetThreshold(gainSettingType gain, double voltage);
void AutomaticGainControl(agcType *agcData);
inline void SetLowGain(gainBitsType gainCtrl);
inline void SetMidGain(gainBitsType gainCtrl);
inline void SetHighGain(gainBitsType gainCtrl);
inline bool GetLowGain(gainBitsType gainCtrl);
inline bool GetMidGain(gainBitsType gainCtrl);
inline bool GetHighGain(gainBitsType gainCtrl);

// Set analog gain to -1.3.
inline void SetLowGain(gainBitsType gainCtrl)
{
  digitalWrite(gainCtrl.g1, LOW);
  digitalWrite(gainCtrl.g0, LOW);
  return;
}

// Set analog gain to -10.2.
inline void SetMidGain(gainBitsType gainCtrl)
{
  digitalWrite(gainCtrl.g1, LOW);
  digitalWrite(gainCtrl.g0, HIGH);
  return;
}

// Set analog gain to -63.
inline void SetHighGain(gainBitsType gainCtrl)
{
  digitalWrite(gainCtrl.g1, HIGH);
  digitalWrite(gainCtrl.g0, HIGH);
  return;
}

// Return true if analog gain is set to -1.3.
inline bool GetLowGain(gainBitsType gainCtrl)
{
  return (digitalRead(gainCtrl.g1) == LOW) && (digitalRead(gainCtrl.g0) == LOW);
}

// Return true if analog gain is set to -10.2
inline bool GetMidGain(gainBitsType gainCtrl)
{
  return (digitalRead(gainCtrl.g1) == LOW) && (digitalRead(gainCtrl.g0) == HIGH);
}

// Return true if analog gain is set to -63.
inline bool GetHighGain(gainBitsType gainCtrl)
{
  return (digitalRead(gainCtrl.g1) == HIGH) && (digitalRead(gainCtrl.g0) == HIGH);
}

// Calculate threshold for switching analog front end gain.
double GetThreshold(gainSettingType gain, double voltage)
{
  double threshold = CT_MAX * voltage; // Max Apparent Power = max mains Irms * mains Vrms

  // Adjust for analog front end gain. 
  switch (gain)
  {
  case lowGain:
    threshold /= -LOW_GAIN;
    break;
  case midGain:
    threshold /= -MID_GAIN;
    break;
  case highGain:
    threshold /= -HIGH_GAIN;
    break;
  default:
    break;
  }

  return threshold;
}

// Get current analog front end gain setings.
gainSettingType GetGainSettings(gainBitsType gainCtrl)
{
  if (GetLowGain(gainCtrl))
  {
    return lowGain;
  }
  else if (GetMidGain(gainCtrl))
  {
    return midGain;
  }
  else if (GetHighGain(gainCtrl))
  {
    return highGain;
  }
  else
  {
    return invalidGain;
  }
}

// Automatic gain control of analog front end.
//
// Apparent power is compared against a threshold to change gain.
// The threshold is set by the maximum apparent power of the current
// transformer and mains voltage. Before a new gain is applied, a check
// is done to ensure the ADC input is not overloaded. This would cause
// incorrect readings, potentially preventing agc loop closure. The current
// channels are recalibrated each time the gain is adjusted. 
void AutomaticGainControl(agcType *agcData)
{
  double vADC;
  double ampCalLow, ampCalMid, ampCalHigh, phaseCal;
  double threshold;
  gainBitsType gainControl;
  gainSettingType gain;
  byte adcInput = agcData->channel + 1;

  if (adcInput == I1_CHAN)
  {
    ampCalLow = I1_AMP_CAL_LOW;
    ampCalMid = I1_AMP_CAL_MID;
    ampCalHigh = I1_AMP_CAL_HIGH;
    phaseCal = I1_PH_CAL;
    gainControl = ct1GainCtrl;
  }
  else if (adcInput == I2_CHAN)
  {
    ampCalLow = I2_AMP_CAL_LOW;
    ampCalMid = I2_AMP_CAL_MID;
    ampCalHigh = I2_AMP_CAL_HIGH;
    phaseCal = I2_PH_CAL;
    gainControl = ct2GainCtrl;
  }
  else
  {
    /* invalid */
  }

  gain = GetGainSettings(gainControl);
  threshold = GetThreshold(gain, agcData->vRMS);

  if (agcData->apparentPower > threshold * UPPER_THRESHOLD)
  {
    // Decrement gains.
    if (gain == highGain)
    {
      SetMidGain(gainControl);
      EmonLibCM_ReCalibrate_IChannel(adcInput, ampCalMid, phaseCal);
    }
    else if (gain == midGain)
    {
      SetLowGain(gainControl);
      EmonLibCM_ReCalibrate_IChannel(adcInput, ampCalLow, phaseCal);
    }
    else
    {/* already at low gain, do nothing */}
  }
  else if (agcData->apparentPower < threshold * LOWER_THRESHOLD)
  {
    // Increment gains.
    if (gain == lowGain)
    {
      vADC = agcData->iRMS / ampCalLow;   // approx rms voltage at ADC input
      if (vADC * -MID_GAIN < ADC_FS_VRMS) // validity check
      {
        SetMidGain(gainControl);
        EmonLibCM_ReCalibrate_IChannel(adcInput, ampCalMid, phaseCal);
      }
    }
    else if (gain == midGain)
    {
      vADC = agcData->iRMS / ampCalMid;     // approx rms voltage at ADC input
      if (vADC * -HIGH_GAIN < ADC_FS_VRMS)  // validity check
      {
        SetHighGain(gainControl);
        EmonLibCM_ReCalibrate_IChannel(adcInput, ampCalHigh, phaseCal);
      }
    }
    else
    {/* already at high gain, do nothing */}
  }
}

void setup()
{
  // Set GPIOs as outputs.
  pinMode(CT1_G0, OUTPUT);
  pinMode(CT1_G1, OUTPUT);
  pinMode(CT2_G0, OUTPUT);
  pinMode(CT2_G1, OUTPUT);

  // Set initial analog front end gains.
  SetLowGain(ct1GainCtrl);
  SetLowGain(ct2GainCtrl);

  Serial.begin(115200);

  EmonLibCM_SetADC_VChannel(V_CHAN, V_AMP_CAL);                   // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(I1_CHAN, I1_AMP_CAL_LOW, I1_PH_CAL);  // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(I2_CHAN, I2_AMP_CAL_LOW, I2_PH_CAL);  // The current channels will be read in this order

  EmonLibCM_setADC_VRef(INTERNAL2V56);                            // ADC Reference voltage (set to 2.56V)
  EmonLibCM_ADCCal(ADC_FS);                                       // ADC Cal voltage (set to 2.56V)
  
  EmonLibCM_cycles_per_second(60);                                // Line frequency (set to 60 Hz)

  EmonLibCM_datalog_period(8.0);                                  // Interval over which stats are reported (in secs)

  EmonLibCM_Init();                                               // Start continuous monitoring
}

void loop()
{
  agcType agcData;
  agcType *agcDataPtr = &agcData;
  double iRMS, vRMS;
  double appPower;

  if (EmonLibCM_Ready())
  {
    vRMS = EmonLibCM_getVrms();
    Serial.print(vRMS);Serial.print(",");

    for (size_t i = 0; i < NUM_ADC_I_CHAN; i++)
    {
      iRMS = EmonLibCM_getIrms(i);
      appPower = EmonLibCM_getApparentPower(i);
     
      Serial.print(iRMS,3);Serial.print(",");
      Serial.print(EmonLibCM_getRealPower(i));Serial.print(",");
      Serial.print(appPower);
      if (i < NUM_ADC_I_CHAN - 1) Serial.print(",");

      // Run automatic gain control of analog front end.
      agcDataPtr->channel = i;
      agcDataPtr->vRMS = vRMS;
      agcDataPtr->iRMS = iRMS;
      agcDataPtr->apparentPower = appPower;
      agc(agcDataPtr);
    }

    Serial.println(); // Outputs one sample.

    delay(50);
  }
}