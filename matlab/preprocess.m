function [] = preprocess()
    startup;
    globals;
    shapenetDir = shapenetDir;
    cachedir = cachedir;

    tsdfDir = fullfile(cachedir, 'shapenet', 'chamferData', '01');

    gridSize = 64;
    numSamples = 10000;
    numShapes = 2000;

    V = niftiread('mito-endolyso.nii');
    V = V == 1;
    V = imerode(V, strel("cube", 6));
    CC = bwconncomp(V);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [sorted,indices] = sort(numPixels, 'descend');

    pBar = TimedProgressBar(numShapes, 30, 'Time Remaining : ', ' Percentage Completion ', 'Tsdf Extraction Completed.');

    for i = 1:numShapes
        X = zeros(size(V));
        X(CC.PixelIdxList{indices(i)}) = 1;
        cropped = regionprops(X, "FilledImage").FilledImage;
        [FV, ~] = Voxel2mesh(cropped);

        maxSize = max(size(cropped));
        normalizedVertices = (FV.vertices - 0.5) / maxSize;
        standardizedVertices = normalizedVertices - 0.5;

        surfaceSamples = uniform_sampling(FV.faces, standardizedVertices, numSamples);
        surfaceSamples = surfaceSamples';

        maxSize = max(size(cropped));
        Volume = imresize3(cropped, gridSize / maxSize);
        padding = gridSize - size(Volume);
        Volume = padarray(Volume, padding, 0, 'post');
        if ~isequal(size(Volume), [gridSize gridSize gridSize])
            disp("!!!!!!!!");
            disp(i);
        end

        stepRange = -0.5 + 1 / (2 * gridSize) :1 / gridSize : 0.5 - 1 / (2 * gridSize);
        [Xp,Yp,Zp] = ndgrid(stepRange, stepRange, stepRange);
        queryPoints = [Xp(:), Yp(:), Zp(:)];

        [tsdfPoints,~,closestPoints] = point_mesh_squared_distance(queryPoints, standardizedVertices, FV.faces);
        tsdfPoints = sqrt(tsdfPoints);
        tsdfGrid = reshape(tsdfPoints, size(Xp));
        tsdfGrid = abs(tsdfGrid) .* (1 - 2 * Volume);
        closestPointsGrid = reshape(closestPoints, [size(Xp), 3]);

        tsdfFile = fullfile(tsdfDir, strcat(num2str(i), '.mat'));
        savefunc(tsdfFile, tsdfGrid, Volume, closestPointsGrid, surfaceSamples, standardizedVertices, FV.faces);
        pBar.progress();
    end

    pBar.stop();
end

function savefunc(tsdfFile, tsdf, Volume, closestPoints, surfaceSamples, vertices, faces)
    save(tsdfFile, 'tsdf', 'Volume', 'closestPoints', 'surfaceSamples', 'vertices', 'faces');
end

function [samples] = uniform_sampling(faces, vertices, numSamples)
    samples = zeros(3, numSamples);
    faceIds = datasample(1:size(faces, 1), numSamples);
    paras = rand(2, numSamples);

    for sId = 1:numSamples
        faceId = faceIds(sId);
        p1 = vertices(faces(faceId, 1), :);
        p2 = vertices(faces(faceId, 2), :);
        p3 = vertices(faces(faceId, 3), :);
        
        r1 = paras(1, sId);
        r2 = paras(2, sId);
        t1 = 1-sqrt(r1);
        t2 = sqrt(r1)*(1-r2);
        t3 = sqrt(r1)*r2;
        samples(:, sId) = t1*p1 + t2*p2 + t3*p3;
    end
end