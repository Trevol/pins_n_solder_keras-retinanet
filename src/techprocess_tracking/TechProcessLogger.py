from techprocess_tracking.TechProcesLogRecord import TechProcesLogRecord


class TechProcessLogger:
    @staticmethod
    def logChanges(logRecord: TechProcesLogRecord):
        return
        framePos = logRecord.framePos
        framePosMs = logRecord.framePosMs
        pinsCount = logRecord.pinsCount
        pinsWithSolderCount = logRecord.pinsWithSolderCount
        logRecord = f'{framePos},{framePosMs:.0f},{pinsCount},{logRecord.pinsAdded},{pinsWithSolderCount},{logRecord.solderAdded}'
        print(logRecord)
