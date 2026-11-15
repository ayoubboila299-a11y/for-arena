local modelID = 281
local txd, dff
local imported = false
local function cleanup()
    if imported then engineRestoreModel(modelID); imported = false end
    if isElement(dff) then destroyElement(dff) end
    if isElement(txd) then destroyElement(txd) end
    dff, txd = nil, nil
end
addEventHandler("onClientResourceStart", resourceRoot, function()
    txd = engineLoadTXD("sfpd1.txd")
    if not txd then outputDebugString("port_commander: TXD load failed", 1); return end
    dff = engineLoadDFF("sfpd1.dff")
    if not dff then outputDebugString("port_commander: DFF load failed", 1); cleanup(); return end
    if not engineImportTXD(txd, modelID) then outputDebugString("port_commander: TXD import failed", 1); cleanup(); return end
    imported = true
    if not engineReplaceModel(dff, modelID) then outputDebugString("port_commander: model replacement failed", 1); cleanup(); return end
end)
addEventHandler("onClientResourceStop", resourceRoot, cleanup)
